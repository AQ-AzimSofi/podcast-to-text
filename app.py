import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import subprocess
import tempfile
import requests
import xml.etree.ElementTree as ET
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_BATCH_MIN = 10


def get_api_keys():
    free = []
    for i in range(1, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            free.append(k)
    paid = os.getenv("GEMINI_API_KEY_PAID", "").strip() or None
    return free, paid


# ---------------------------------------------------------------------------
# Podcast download
# ---------------------------------------------------------------------------

def resolve_podcast_audio(apple_url):
    match = re.search(r"/id(\d+)", apple_url)
    if not match:
        raise ValueError("Apple Podcast URLからIDを取得できませんでした")
    podcast_id = match.group(1)

    episode_match = re.search(r"[?&]i=(\d+)", apple_url)
    episode_id = episode_match.group(1) if episode_match else None

    lookup = requests.get(
        f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast",
        timeout=15,
    ).json()
    feed_url = lookup["results"][0]["feedUrl"]

    rss = requests.get(feed_url, timeout=30).text
    root = ET.fromstring(rss)

    ns = {
        "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
        "podcast": "https://podcastindex.org/namespace/1.0",
    }

    for item in root.iter("item"):
        if episode_id:
            guid = item.findtext("guid", "")
            ep_url_candidates = []
            enc = item.find("enclosure")
            if enc is not None:
                ep_url_candidates.append(enc.get("url", ""))
            ep_match = any(episode_id in c for c in [guid] + ep_url_candidates)
            if not ep_match:
                itunes_ep = item.find("itunes:episode", ns)
                if itunes_ep is not None and itunes_ep.text:
                    pass
            if enc is not None:
                audio_url = enc.get("url")
                if episode_id and not ep_match:
                    continue
                title = item.findtext("title", "episode")
                return audio_url, title
        else:
            enc = item.find("enclosure")
            if enc is not None:
                title = item.findtext("title", "episode")
                return enc.get("url"), title

    if not episode_id:
        raise ValueError("RSSフィードにエピソードが見つかりません")

    for item in root.iter("item"):
        enc = item.find("enclosure")
        if enc is not None:
            audio_url = enc.get("url")
            resp = requests.head(audio_url, allow_redirects=True, timeout=10)
            if episode_id in resp.url or episode_id in audio_url:
                title = item.findtext("title", "episode")
                return audio_url, title

    first = root.find(".//item/enclosure")
    if first is not None:
        title = root.findtext(".//item/title", "episode")
        return first.get("url"), title

    raise ValueError("音声ファイルURLが見つかりません")


def download_audio(url, dest_path, progress_cb=None):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_cb and total:
                progress_cb(downloaded / total)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def split_audio_segment(audio_path, start_min, end_min):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start_min * 60),
        "-to", str(end_min * 60),
        "-c", "copy", tmp.name,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return tmp.name


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_TEXT = (
    "この音声を日本語で文字起こししてください。"
    "話者が複数いる場合は話者を区別してください。"
    "タイムスタンプは不要です。"
)

PROMPT_SRT = (
    "この音声を日本語で文字起こしし、SRT字幕形式で出力してください。\n"
    "ルール:\n"
    "- 1つの字幕は最大40文字程度にしてください。長い発話は複数の字幕に分割してください。\n"
    "- 1つの字幕の時間は最大10秒以内にしてください。\n"
    "- タイムスタンプは必ず HH:MM:SS,mmm 形式（例: 00:01:02,500）で書いてください。HH:は省略しないでください。\n"
    "- 話者が複数いる場合は行頭に話者名を付けてください。\n"
    "- 日本語テキストに不要なスペースを入れないでください。自然な日本語表記にしてください。\n"
    "- SRT形式以外のテキストは出力しないでください。\n\n"
    "出力例:\n"
    "1\n"
    "00:00:00,000 --> 00:00:03,500\n"
    "話者A: こんにちは、今日はよろしくお願いします。\n\n"
    "2\n"
    "00:00:03,500 --> 00:00:07,200\n"
    "話者B: よろしくお願いします。\n"
)


# ---------------------------------------------------------------------------
# SRT post-processing
# ---------------------------------------------------------------------------

def _normalize_ts(ts):
    ts = ts.strip()
    if "," in ts:
        time_part, frac = ts.rsplit(",", 1)
    elif "." in ts:
        time_part, frac = ts.rsplit(".", 1)
    else:
        time_part, frac = ts, "000"

    frac = frac.ljust(3, "0")[:3]
    segs = time_part.split(":")

    if len(segs) == 2:
        a, b = int(segs[0]), int(segs[1])
        h, m, s = 0, a, b
    elif len(segs) == 3:
        h, m, s = int(segs[0]), int(segs[1]), int(segs[2])
    else:
        h, m, s = 0, 0, 0

    if s >= 100:
        frac = str(s).ljust(3, "0")[:3]
        s = 0
    elif s >= 60:
        m += s // 60
        s = s % 60
    if m >= 60:
        h += m // 60
        m = m % 60

    return f"{h:02d}:{m:02d}:{s:02d},{frac}"


def _ts_to_ms(ts):
    ts = _normalize_ts(ts)
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def _ms_to_ts(ms):
    ms = max(0, ms)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


MAX_CHARS = 80


def _remove_jp_spaces(text):
    JP = r"\u3000-\u9FFF\uF900-\uFAFF"
    PUNCT = r"\u3000-\u303F\uFF00-\uFFEF"
    text = re.sub(rf"(\S) +([{PUNCT}])", r"\1\2", text)
    text = re.sub(rf"([{PUNCT}]) +(\S)", r"\1\2", text)
    text = re.sub(rf"([{JP}]) +([{JP}])", r"\1\2", text)
    text = re.sub(rf"([{JP}]) +([{JP}])", r"\1\2", text)
    return text


def _try_reinterpret_ts(ts_str, target_ms):
    h, m, s_ms = ts_str.split(":")
    s, ms = s_ms.split(",")
    h, m, s, ms = int(h), int(m), int(s), int(ms)
    original_ms = h * 3600000 + m * 60000 + s * 1000 + ms

    candidates = [original_ms]
    candidates.append(h * 60000 + m * 1000 + int(f"{s}{ms:03d}"[:3]))
    candidates.append(m * 60000 + s * 1000 + ms)
    raw_digits = f"{h:02d}{m:02d}{s:02d}"
    for split_pt in range(1, len(raw_digits)):
        try:
            mins = int(raw_digits[:split_pt])
            secs = int(raw_digits[split_pt:split_pt + 2]) if split_pt + 2 <= len(raw_digits) else 0
            if mins < 60 and secs < 60:
                candidates.append(mins * 60000 + secs * 1000 + ms)
        except ValueError:
            continue

    best = min(candidates, key=lambda c: abs(c - target_ms) if c >= 0 else float("inf"))
    return best


def _fix_timestamps(entries):
    if not entries:
        return entries
    fixed = []
    prev_end_ms = 0
    for start, end, text in entries:
        start_ms = _ts_to_ms(start)
        end_ms = _ts_to_ms(end)

        if abs(start_ms - prev_end_ms) > 60000:
            start_ms = _try_reinterpret_ts(start, prev_end_ms)
            if abs(start_ms - prev_end_ms) > 60000:
                start_ms = prev_end_ms
            start = _ms_to_ts(start_ms)

        end_ms = _ts_to_ms(end)
        if end_ms - start_ms > 60000 or end_ms < start_ms:
            end_ms = _try_reinterpret_ts(end, start_ms + 5000)
            if end_ms - start_ms > 60000 or end_ms < start_ms:
                end_ms = start_ms + 5000
            end = _ms_to_ts(end_ms)

        if end_ms <= start_ms:
            end_ms = start_ms + 2000
            end = _ms_to_ts(end_ms)

        prev_end_ms = end_ms
        fixed.append((start, end, text))
    return fixed


def _fix_srt(raw):
    blocks = re.split(r"\n\s*\n", raw.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        ts_line = lines[1] if "-->" in lines[1] else None
        if not ts_line:
            continue
        text = "\n".join(lines[2:]).strip()
        text = _remove_jp_spaces(text)
        start_raw, end_raw = ts_line.split("-->")
        start = _normalize_ts(start_raw)
        end = _normalize_ts(end_raw)
        entries.append((start, end, text))

    entries = _fix_timestamps(entries)

    result = []
    idx = 1
    for start, end, text in entries:
        if len(text) <= MAX_CHARS:
            result.append(f"{idx}\n{start} --> {end}\n{text}")
            idx += 1
            continue

        chunks = re.split(r"(?<=[。、！？\.\!\?])", text)
        merged = []
        buf = ""
        for c in chunks:
            if not c:
                continue
            if len(buf) + len(c) <= MAX_CHARS:
                buf += c
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)

        if any(len(m) > MAX_CHARS for m in merged):
            final = []
            for m in merged:
                while len(m) > MAX_CHARS:
                    final.append(m[:MAX_CHARS])
                    m = m[MAX_CHARS:]
                if m:
                    final.append(m)
            merged = final

        start_ms = _ts_to_ms(start)
        end_ms = _ts_to_ms(end)
        duration = end_ms - start_ms
        total_chars = sum(len(m) for m in merged)

        offset = 0
        for j, chunk in enumerate(merged):
            frac_start = offset / total_chars if total_chars else 0
            offset += len(chunk)
            frac_end = offset / total_chars if total_chars else 1
            c_start = _ms_to_ts(start_ms + int(duration * frac_start))
            c_end = _ms_to_ts(start_ms + int(duration * frac_end))
            result.append(f"{idx}\n{c_start} --> {c_end}\n{chunk}")
            idx += 1

    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# SRT offset / combine
# ---------------------------------------------------------------------------

def offset_srt(srt_text, offset_ms):
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    result = []
    idx = 1
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = lines[1].split("-->")
        start_ms = _ts_to_ms(start_raw.strip()) + offset_ms
        end_ms = _ts_to_ms(end_raw.strip()) + offset_ms
        text = "\n".join(lines[2:])
        result.append(f"{idx}\n{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}\n{text}")
        idx += 1
    return "\n\n".join(result)


def srt_time_range(srt_text):
    if not srt_text or not srt_text.strip():
        return None, None
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    first_ms = None
    last_ms = None
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = lines[1].split("-->")
        s = _ts_to_ms(start_raw.strip())
        e = _ts_to_ms(end_raw.strip())
        if first_ms is None:
            first_ms = s
        last_ms = e
    return first_ms, last_ms


def trim_srt(srt_text, keep_before_ms=None, keep_after_ms=None):
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    result = []
    idx = 1
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = lines[1].split("-->")
        s = _ts_to_ms(start_raw.strip())
        if keep_before_ms is not None and s >= keep_before_ms:
            continue
        if keep_after_ms is not None and s < keep_after_ms:
            continue
        ts_and_text = "\n".join(lines[1:])
        result.append(f"{idx}\n{ts_and_text}")
        idx += 1
    return "\n\n".join(result)


def combine_srt(*srt_parts):
    result = []
    idx = 1
    for part in srt_parts:
        if not part or not part.strip():
            continue
        blocks = re.split(r"\n\s*\n", part.strip())
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 3 or "-->" not in lines[1]:
                continue
            ts_and_text = "\n".join(lines[1:])
            result.append(f"{idx}\n{ts_and_text}")
            idx += 1
    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# Gemini transcription
# ---------------------------------------------------------------------------

def _transcribe_single(audio_path, api_key, srt_mode, log_cb=None):
    prompt = PROMPT_SRT if srt_mode else PROMPT_TEXT
    client = genai.Client(api_key=api_key)
    file = client.files.upload(file=audio_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_uri(
                        file_uri=file.uri,
                        mime_type=file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
    )
    text = response.text
    if srt_mode:
        text = text.strip().removeprefix("```srt").removeprefix("```").removesuffix("```").strip()
        text = _fix_srt(text)
    return text


def transcribe_with_fallback(audio_path, free_keys, paid_key, srt_mode, log_cb=None):
    keys_to_try = list(free_keys)
    if paid_key:
        keys_to_try.append(paid_key)
    for i, key in enumerate(keys_to_try):
        is_paid = (paid_key and key == paid_key)
        label = "PAID" if is_paid else f"FREE-{i+1}"
        if log_cb:
            log_cb(f"  key {label} を試行中...")
        try:
            text = _transcribe_single(audio_path, key, srt_mode, log_cb)
            if log_cb:
                log_cb(f"  key {label} 成功")
            return text
        except Exception as e:
            if log_cb:
                log_cb(f"  key {label} 失敗: {e}")
            if i == len(keys_to_try) - 1:
                raise
    raise RuntimeError("No API keys available")


def transcribe_parallel(audio_path, free_keys, paid_key, srt_mode,
                        range_start_min, range_end_min, batch_min, log_cb=None):
    segments = []
    t = range_start_min
    while t < range_end_min:
        seg_end = min(t + batch_min, range_end_min)
        segments.append((t, seg_end))
        t = seg_end

    if log_cb:
        log_cb(f"Parallel: {len(segments)} segments, {len(free_keys)} free keys")

    tmp_files = []
    for start, end in segments:
        if start == 0 and end * 60 >= get_audio_duration(audio_path):
            tmp_files.append(audio_path)
        else:
            tmp_files.append(split_audio_segment(audio_path, start, end))

    results = [None] * len(segments)
    errors = [None] * len(segments)

    def worker(seg_idx, audio_file, assigned_key):
        start_min, end_min = segments[seg_idx]
        tag = f"[{start_min:02d}:00-{end_min:02d}:00]"
        if log_cb:
            log_cb(f"{tag} transcribing...")
        try:
            text = _transcribe_single(audio_file, assigned_key, srt_mode, log_cb)
            results[seg_idx] = text
            if log_cb:
                log_cb(f"{tag} done")
        except Exception as e:
            if log_cb:
                log_cb(f"{tag} key failed: {e}, trying fallback...")
            remaining_keys = [k for k in (free_keys + ([paid_key] if paid_key else []))
                              if k != assigned_key]
            for fallback in remaining_keys:
                try:
                    text = _transcribe_single(audio_file, fallback, srt_mode, log_cb)
                    results[seg_idx] = text
                    if log_cb:
                        log_cb(f"{tag} fallback succeeded")
                    return
                except Exception:
                    continue
            errors[seg_idx] = str(e)
            if log_cb:
                log_cb(f"{tag} all keys failed")

    with ThreadPoolExecutor(max_workers=len(free_keys) or 1) as pool:
        futures = []
        for i in range(len(segments)):
            key = free_keys[i % len(free_keys)] if free_keys else paid_key
            futures.append(pool.submit(worker, i, tmp_files[i], key))
        for f in as_completed(futures):
            f.result()

    for tmp in tmp_files:
        if tmp != audio_path and os.path.exists(tmp):
            os.unlink(tmp)

    failed = [i for i, e in enumerate(errors) if e is not None]
    if failed:
        labels = [f"{segments[i][0]:02d}:00-{segments[i][1]:02d}:00" for i in failed]
        raise RuntimeError(f"Failed segments: {', '.join(labels)}")

    if srt_mode:
        offset_results = []
        for i, (start_min, _) in enumerate(segments):
            text = results[i]
            if start_min > 0:
                text = offset_srt(text, start_min * 60 * 1000)
            offset_results.append(text)
        return combine_srt(*offset_results)
    else:
        return "\n\n".join(r for r in results if r)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Podcast to Text")
        self.geometry("780x650")
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- URL input ---
        url_frame = ttk.LabelFrame(self, text="1. Podcast URL")
        url_frame.pack(fill="x", **pad)

        self.url_var = tk.StringVar()
        ttk.Entry(url_frame, textvariable=self.url_var).pack(
            side="left", fill="x", expand=True, **pad
        )
        self.dl_btn = ttk.Button(
            url_frame, text="Download MP3", command=self._on_download
        )
        self.dl_btn.pack(side="right", **pad)

        # --- or pick local file ---
        file_frame = ttk.LabelFrame(self, text="or Local MP3")
        file_frame.pack(fill="x", **pad)

        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(
            side="left", fill="x", expand=True, **pad
        )
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(
            side="right", **pad
        )

        # --- range & parallel ---
        range_frame = ttk.LabelFrame(self, text="2. Range & Mode")
        range_frame.pack(fill="x", **pad)

        ttk.Label(range_frame, text="From (min):").pack(side="left", **pad)
        self.range_from = tk.IntVar(value=0)
        ttk.Spinbox(range_frame, from_=0, to=999, width=5,
                     textvariable=self.range_from).pack(side="left", **pad)

        ttk.Label(range_frame, text="To (min):").pack(side="left", **pad)
        self.range_to = tk.IntVar(value=0)
        self.range_to_spin = ttk.Spinbox(range_frame, from_=0, to=999, width=5,
                                          textvariable=self.range_to)
        self.range_to_spin.pack(side="left", **pad)

        ttk.Button(range_frame, text="Detect", command=self._on_detect).pack(
            side="left", **pad
        )

        ttk.Separator(range_frame, orient="vertical").pack(side="left", fill="y", padx=4)

        ttk.Label(range_frame, text="Batch (min):").pack(side="left", **pad)
        self.batch_min = tk.IntVar(value=DEFAULT_BATCH_MIN)
        ttk.Spinbox(range_frame, from_=1, to=60, width=4,
                     textvariable=self.batch_min).pack(side="left", **pad)

        self.parallel_var = tk.BooleanVar()
        ttk.Checkbutton(
            range_frame, text="Parallel", variable=self.parallel_var
        ).pack(side="left", **pad)

        # --- previous SRT ---
        srt_import_frame = ttk.LabelFrame(self, text="Import previous SRT (optional, for combining)")
        srt_import_frame.pack(fill="x", **pad)

        self.prev_srt_var = tk.StringVar()
        ttk.Entry(srt_import_frame, textvariable=self.prev_srt_var).pack(
            side="left", fill="x", expand=True, **pad
        )
        ttk.Button(srt_import_frame, text="Browse",
                    command=self._browse_prev_srt).pack(side="right", **pad)
        ttk.Button(srt_import_frame, text="Clear",
                    command=self._clear_prev_srt).pack(side="right", **pad)

        self.prev_srt_info = tk.StringVar(value="")
        ttk.Label(srt_import_frame, textvariable=self.prev_srt_info,
                  foreground="blue").pack(side="left", **pad)

        # --- progress ---
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", **pad)

        # --- transcribe ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **pad)

        self.srt_var = tk.BooleanVar()
        ttk.Checkbutton(
            action_frame, text="SRT format", variable=self.srt_var
        ).pack(side="left", **pad)

        self.transcribe_btn = ttk.Button(
            action_frame, text="Transcribe", command=self._on_transcribe
        )
        self.transcribe_btn.pack(side="left", **pad)

        self.save_btn = ttk.Button(
            action_frame, text="Save", command=self._on_save, state="disabled"
        )
        self.save_btn.pack(side="left", **pad)

        # --- log / output ---
        self.log_widget = scrolledtext.ScrolledText(self, height=8, state="disabled")
        self.log_widget.pack(fill="x", **pad)

        ttk.Label(self, text="Transcript:").pack(anchor="w", **pad)
        self.output = scrolledtext.ScrolledText(self, wrap="word")
        self.output.pack(fill="both", expand=True, **pad)

    def _log(self, msg):
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", msg + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _set_progress(self, frac):
        self.progress["value"] = frac * 100

    def _browse_file(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.m4a *.wav")])
        if p:
            self.file_var.set(p)

    def _browse_prev_srt(self):
        p = filedialog.askopenfilename(filetypes=[("SRT", "*.srt")])
        if p:
            self.prev_srt_var.set(p)
            try:
                text = Path(p).read_text(encoding="utf-8")
                first_ms, last_ms = srt_time_range(text)
                if first_ms is not None and last_ms is not None:
                    n_entries = len([b for b in re.split(r"\n\s*\n", text.strip())
                                    if "-->" in b])
                    self.prev_srt_info.set(
                        f"Covers {_ms_to_ts(first_ms)[:8]} - {_ms_to_ts(last_ms)[:8]}  "
                        f"({n_entries} entries)"
                    )
                else:
                    self.prev_srt_info.set("(empty or invalid SRT)")
            except Exception:
                self.prev_srt_info.set("(could not read file)")

    def _clear_prev_srt(self):
        self.prev_srt_var.set("")
        self.prev_srt_info.set("")

    def _on_detect(self):
        audio = self.file_var.get().strip()
        if not audio or not os.path.isfile(audio):
            messagebox.showwarning("Input", "MP3ファイルを先に選択してください")
            return
        try:
            dur = get_audio_duration(audio)
            total_min = int(dur / 60) + 1
            self.range_from.set(0)
            self.range_to.set(total_min)
            self._log(f"Audio: {dur:.0f}s ({dur/60:.1f}min) | Range set to 0-{total_min}")
        except Exception as e:
            self._log(f"Detect error: {e}")

    def _on_download(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Input", "URLを入力してください")
            return
        self.dl_btn.configure(state="disabled")
        threading.Thread(target=self._download_thread, args=(url,), daemon=True).start()

    def _download_thread(self, url):
        try:
            self.after(0, self._log, "Podcast情報を取得中...")
            audio_url, title = resolve_podcast_audio(url)
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:80]
            dest = str(Path(__file__).parent / f"{safe_title}.mp3")
            self.after(0, self._log, f"ダウンロード中: {title}")
            download_audio(
                audio_url, dest,
                progress_cb=lambda f: self.after(0, self._set_progress, f),
            )
            self.after(0, self.file_var.set, dest)
            self.after(0, self._log, f"保存完了: {dest}")
            self.after(0, self._set_progress, 1.0)
        except Exception as e:
            self.after(0, self._log, f"エラー: {e}")
        finally:
            self.after(0, lambda: self.dl_btn.configure(state="normal"))

    def _on_transcribe(self):
        audio = self.file_var.get().strip()
        if not audio or not os.path.isfile(audio):
            messagebox.showwarning("Input", "MP3ファイルを選択またはダウンロードしてください")
            return
        free_keys, paid_key = get_api_keys()
        if not free_keys and not paid_key:
            messagebox.showwarning("API Key", ".envファイルにGemini APIキーを設定してください")
            return
        self.transcribe_btn.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(15)

        opts = {
            "audio": audio,
            "free_keys": free_keys,
            "paid_key": paid_key,
            "srt_mode": self.srt_var.get(),
            "parallel": self.parallel_var.get(),
            "range_from": self.range_from.get(),
            "range_to": self.range_to.get(),
            "batch_min": self.batch_min.get(),
            "prev_srt": self.prev_srt_var.get().strip(),
        }
        threading.Thread(target=self._transcribe_thread, args=(opts,), daemon=True).start()

    def _transcribe_thread(self, opts):
        tmp_audio = None
        try:
            log = lambda m: self.after(0, self._log, m)
            audio = opts["audio"]
            free_keys = opts["free_keys"]
            paid_key = opts["paid_key"]
            srt_mode = opts["srt_mode"]
            r_from = opts["range_from"]
            r_to = opts["range_to"]

            if r_to <= r_from:
                dur = get_audio_duration(audio)
                r_to = int(dur / 60) + 1
                log(f"Auto-detected range: 0-{r_to} min")

            is_full = (r_from == 0 and r_to * 60 >= get_audio_duration(audio) - 5)

            if opts["parallel"] and free_keys:
                log(f"Parallel mode: {r_from}-{r_to} min, batch={opts['batch_min']}min, "
                    f"{len(free_keys)} free keys" +
                    (" + 1 paid key" if paid_key else ""))
                text = transcribe_parallel(
                    audio, free_keys, paid_key, srt_mode,
                    r_from, r_to, opts["batch_min"], log_cb=log,
                )
            else:
                if not is_full:
                    log(f"Extracting segment: {r_from}:00 - {r_to}:00")
                    tmp_audio = split_audio_segment(audio, r_from, r_to)
                    actual = tmp_audio
                else:
                    actual = audio

                all_keys = free_keys + ([paid_key] if paid_key else [])
                text = transcribe_with_fallback(
                    actual, free_keys, paid_key, srt_mode, log_cb=log,
                )

                if srt_mode and r_from > 0:
                    log(f"Offsetting timestamps by +{r_from}:00...")
                    text = offset_srt(text, r_from * 60 * 1000)

            prev_srt = opts["prev_srt"]
            if srt_mode and prev_srt and os.path.isfile(prev_srt):
                prev_text = Path(prev_srt).read_text(encoding="utf-8")
                new_start_ms = r_from * 60 * 1000
                new_end_ms = r_to * 60 * 1000
                prev_first, prev_last = srt_time_range(prev_text)
                if prev_first is not None and prev_last is not None:
                    if prev_first < new_end_ms and prev_last > new_start_ms:
                        overlap_start = _ms_to_ts(max(prev_first, new_start_ms))[:8]
                        overlap_end = _ms_to_ts(min(prev_last, new_end_ms))[:8]
                        log(f"Overlap detected ({overlap_start}-{overlap_end}), "
                            f"trimming previous SRT to keep only entries before {_ms_to_ts(new_start_ms)[:8]}")
                        prev_text = trim_srt(prev_text, keep_before_ms=new_start_ms)
                    else:
                        log("No overlap, combining as-is")
                log(f"Combining with previous SRT...")
                text = combine_srt(prev_text, text)

            self.after(0, self._show_transcript, text)
        except Exception as e:
            self.after(0, self._log, f"エラー: {e}")
        finally:
            if tmp_audio and os.path.exists(tmp_audio):
                os.unlink(tmp_audio)
            self.after(0, self._finish_transcribe)

    def _show_transcript(self, text):
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.save_btn.configure(state="normal")

    def _finish_transcribe(self):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self._set_progress(0)
        self.transcribe_btn.configure(state="normal")

    def _on_save(self):
        text = self.output.get("1.0", "end").strip()
        if not text:
            return
        if self.srt_var.get():
            default_ext, ftypes = ".srt", [("SRT", "*.srt"), ("Text", "*.txt")]
        else:
            default_ext, ftypes = ".txt", [("Text", "*.txt"), ("SRT", "*.srt")]
        p = filedialog.asksaveasfilename(
            defaultextension=default_ext, filetypes=ftypes
        )
        if p:
            Path(p).write_text(text, encoding="utf-8")
            self._log(f"保存しました: {p}")


if __name__ == "__main__":
    App().mainloop()
