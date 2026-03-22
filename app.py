import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import subprocess
import tempfile
import requests
import xml.etree.ElementTree as ET
import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

SEGMENT_MINUTES = 10

load_dotenv()

def get_api_keys():
    keys = []
    for i in range(1, 5):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    paid = os.getenv("GEMINI_API_KEY_PAID", "").strip()
    if paid:
        keys.append(paid)
    return keys


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


def _normalize_ts(ts):
    ts = ts.strip()

    # Split on comma or dot to separate fractional part
    if "," in ts:
        time_part, frac = ts.rsplit(",", 1)
    elif "." in ts:
        time_part, frac = ts.rsplit(".", 1)
    else:
        time_part, frac = ts, "000"

    frac = frac.ljust(3, "0")[:3]
    segs = time_part.split(":")

    if len(segs) == 2:
        # Could be MM:SS or HH:MM — check if second value > 59 means it's not seconds
        a, b = int(segs[0]), int(segs[1])
        # MM:SS format (missing HH:)
        h, m, s = 0, a, b
    elif len(segs) == 3:
        h, m, s = int(segs[0]), int(segs[1]), int(segs[2])
    else:
        h, m, s = 0, 0, 0

    # Fix overflows: if s >= 60, it was probably ms packed into s field
    # e.g. "01:02:790" where 790 is actually ,790 not seconds
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
    # Remove space before Japanese punctuation: "word 。" -> "word。"
    text = re.sub(rf"(\S) +([{PUNCT}])", r"\1\2", text)
    # Remove space after Japanese punctuation: "。 word" -> "。word"
    text = re.sub(rf"([{PUNCT}]) +(\S)", r"\1\2", text)
    # Remove spaces between Japanese characters: "漢字 漢字" -> "漢字漢字"
    text = re.sub(rf"([{JP}]) +([{JP}])", r"\1\2", text)
    # Repeat to catch chains: "あ い う" needs multiple passes
    text = re.sub(rf"([{JP}]) +([{JP}])", r"\1\2", text)
    return text


def _try_reinterpret_ts(ts_str, target_ms):
    h, m, s_ms = ts_str.split(":")
    s, ms = s_ms.split(",")
    h, m, s, ms = int(h), int(m), int(s), int(ms)
    original_ms = h * 3600000 + m * 60000 + s * 1000 + ms

    candidates = [original_ms]
    # HH:MM:SS,mmm -> treat HH as MM, MM as SS, SS+mmm as mmm
    candidates.append(h * 60000 + m * 1000 + int(f"{s}{ms:03d}"[:3]))
    # HH:MM:SS,mmm -> just zero out HH (00:MM:SS,mmm)
    candidates.append(m * 60000 + s * 1000 + ms)
    # Digits got concatenated: e.g. 18:52:03,496 was meant to be 00:18:52,034
    # Try reading "HH" + first digit of MM as minutes, rest as seconds
    raw_digits = f"{h:02d}{m:02d}{s:02d}"
    for split_pt in range(1, len(raw_digits)):
        try:
            mins = int(raw_digits[:split_pt])
            secs = int(raw_digits[split_pt:split_pt+2]) if split_pt + 2 <= len(raw_digits) else 0
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


def combine_srt(existing_srt, new_srt):
    blocks_a = re.split(r"\n\s*\n", existing_srt.strip()) if existing_srt.strip() else []
    blocks_b = re.split(r"\n\s*\n", new_srt.strip()) if new_srt.strip() else []
    all_blocks = blocks_a + blocks_b
    result = []
    idx = 1
    for block in all_blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        ts_and_text = "\n".join(lines[1:])
        result.append(f"{idx}\n{ts_and_text}")
        idx += 1
    return "\n\n".join(result)


def transcribe_with_gemini(audio_path, api_keys, srt_mode=False, log_cb=None):
    prompt = PROMPT_SRT if srt_mode else PROMPT_TEXT
    for i, key in enumerate(api_keys):
        if log_cb:
            log_cb(f"APIキー {i+1}/{len(api_keys)} を試行中...")
        try:
            client = genai.Client(api_key=key)
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
            if log_cb:
                log_cb(f"APIキー {i+1} で成功")
            text = response.text
            if srt_mode:
                text = text.strip().removeprefix("```srt").removeprefix("```").removesuffix("```").strip()
                if log_cb:
                    log_cb("SRT後処理中（タイムスタンプ修正 + 長文分割）...")
                text = _fix_srt(text)
            return text
        except Exception as e:
            err = str(e)
            if log_cb:
                log_cb(f"APIキー {i+1} 失敗: {err}")
            if i == len(api_keys) - 1:
                raise RuntimeError(f"全てのAPIキーが失敗しました。最後のエラー: {err}")
            continue


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Podcast to Text")
        self.geometry("750x600")
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

        # --- segment selection ---
        seg_frame = ttk.LabelFrame(self, text="Segment")
        seg_frame.pack(fill="x", **pad)

        self.segment_var = tk.StringVar(value="Full")
        self.segment_combo = ttk.Combobox(
            seg_frame, textvariable=self.segment_var,
            values=["Full"], state="readonly", width=20,
        )
        self.segment_combo.pack(side="left", **pad)

        ttk.Button(
            seg_frame, text="Detect segments", command=self._on_detect_segments
        ).pack(side="left", **pad)

        ttk.Separator(seg_frame, orient="vertical").pack(side="left", fill="y", padx=4)

        self.prev_srt_var = tk.StringVar()
        ttk.Label(seg_frame, text="Previous SRT:").pack(side="left", **pad)
        ttk.Entry(seg_frame, textvariable=self.prev_srt_var, width=30).pack(
            side="left", fill="x", expand=True, **pad
        )
        ttk.Button(
            seg_frame, text="Browse", command=self._browse_prev_srt
        ).pack(side="right", **pad)

        # --- progress ---
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", **pad)

        # --- transcribe ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **pad)

        self.srt_var = tk.BooleanVar()
        ttk.Checkbutton(
            action_frame, text="SRT format (with timestamps)", variable=self.srt_var
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
        self.log = scrolledtext.ScrolledText(self, height=6, state="disabled")
        self.log.pack(fill="x", **pad)

        ttk.Label(self, text="Transcript:").pack(anchor="w", **pad)
        self.output = scrolledtext.ScrolledText(self, wrap="word")
        self.output.pack(fill="both", expand=True, **pad)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

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

    def _on_detect_segments(self):
        audio = self.file_var.get().strip()
        if not audio or not os.path.isfile(audio):
            messagebox.showwarning("Input", "MP3ファイルを先に選択してください")
            return
        try:
            dur = get_audio_duration(audio)
            total_min = dur / 60
            segs = ["Full"]
            start = 0
            while start < total_min:
                end = min(start + SEGMENT_MINUTES, int(total_min) + 1)
                segs.append(f"{start:02d}:00 - {end:02d}:00")
                start += SEGMENT_MINUTES
            self.segment_combo["values"] = segs
            self.segment_var.set("Full")
            self._log(f"Audio: {dur:.0f}s ({total_min:.1f}min) - {len(segs)-1} segments")
        except Exception as e:
            self._log(f"Segment detection error: {e}")

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
                audio_url,
                dest,
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
        keys = get_api_keys()
        if not keys or keys[0].startswith("your-key"):
            messagebox.showwarning("API Key", ".envファイルにGemini APIキーを設定してください")
            return
        self.transcribe_btn.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(15)
        srt_mode = self.srt_var.get()
        segment = self.segment_var.get()
        prev_srt_path = self.prev_srt_var.get().strip()
        threading.Thread(
            target=self._transcribe_thread,
            args=(audio, keys, srt_mode, segment, prev_srt_path),
            daemon=True,
        ).start()

    def _transcribe_thread(self, audio_path, keys, srt_mode, segment, prev_srt_path):
        tmp_audio = None
        try:
            log = lambda m: self.after(0, self._log, m)
            start_min = 0
            actual_path = audio_path

            if segment != "Full":
                match = re.match(r"(\d+):00 - (\d+):00", segment)
                if match:
                    start_min = int(match.group(1))
                    end_min = int(match.group(2))
                    log(f"Audio segment: {start_min}:00 - {end_min}:00")
                    tmp_audio = split_audio_segment(audio_path, start_min, end_min)
                    actual_path = tmp_audio

            text = transcribe_with_gemini(
                actual_path, keys, srt_mode=srt_mode, log_cb=log,
            )

            if srt_mode and start_min > 0:
                log(f"Offsetting timestamps by +{start_min}:00...")
                text = offset_srt(text, start_min * 60 * 1000)

            if srt_mode and prev_srt_path and os.path.isfile(prev_srt_path):
                log(f"Combining with previous SRT: {prev_srt_path}")
                prev_srt = Path(prev_srt_path).read_text(encoding="utf-8")
                text = combine_srt(prev_srt, text)

            self.after(0, self._show_transcript, text)
        except Exception as e:
            self.after(0, self._log, f"文字起こしエラー: {e}")
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
