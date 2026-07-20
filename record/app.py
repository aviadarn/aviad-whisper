"""Local voice-recording server for collecting Whisper training data.

Serves a page that shows each generated prompt; you read it aloud and record.
Each clip is transcoded to 16 kHz mono WAV (ffmpeg) and paired with its exact
reference text in an `audiofolder`-compatible metadata.csv the training
pipeline can load directly.

    python record/app.py            # then open http://localhost:8000
    python record/app.py --port 8080

Output:
    data/voice/clip_0001.wav ...
    data/voice/metadata.csv   (columns: file_name, transcription)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "data", "voice")
META = os.path.join(OUT_DIR, "metadata.csv")
PROMPTS = os.path.join(HERE, "prompts.json")

os.makedirs(OUT_DIR, exist_ok=True)

# id -> {"file_name": str, "transcription": str}. Loaded at startup so re-records overwrite.
records: dict[int, dict[str, str]] = {}


def load_existing() -> None:
    if not os.path.exists(META):
        return
    with open(META, newline="") as f:
        for row in csv.DictReader(f):
            fn = row["file_name"]
            try:
                idx = int(os.path.splitext(fn)[0].split("_")[-1])
            except ValueError:
                continue
            records[idx] = {"file_name": fn, "transcription": row["transcription"]}


def write_metadata() -> None:
    with open(META, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "transcription"])
        for idx in sorted(records):
            w.writerow([records[idx]["file_name"], records[idx]["transcription"]])


def transcode(raw: bytes, wav_path: str) -> None:
    """webm/opus (or anything ffmpeg reads) -> 16 kHz mono wav."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp_path)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quieter console
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/prompts.json":
            with open(PROMPTS, "rb") as f:
                self._send(200, f.read(), "application/json")
        elif path == "/status":
            done = sorted(records.keys())
            self._send(200, json.dumps({"done": done, "count": len(done)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path)
        if path.path != "/save":
            self._send(404, b"not found", "text/plain")
            return
        q = parse_qs(path.query)
        try:
            idx = int(q["id"][0])
            text = q["text"][0]
        except (KeyError, ValueError):
            self._send(400, b"missing id/text", "text/plain")
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if not raw:
            self._send(400, b"empty audio", "text/plain")
            return

        fn = f"clip_{idx:04d}.wav"
        wav_path = os.path.join(OUT_DIR, fn)
        try:
            transcode(raw, wav_path)
        except subprocess.CalledProcessError as e:
            self._send(500, b"ffmpeg failed: " + e.stderr[-400:], "text/plain")
            return

        records[idx] = {"file_name": fn, "transcription": text}
        write_metadata()
        self._send(200, json.dumps({"ok": True, "count": len(records)}).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    load_existing()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Recorder at http://localhost:{args.port}  ({len(records)} clips already saved)")
    print(f"Writing to {OUT_DIR}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
