import argparse
import json
import re
import uuid
import subprocess
from pathlib import Path
import sys

#paths config
ROOT = Path(".").resolve()
DATA_ROOT = ROOT / "data" / "gtzan"
ART = ROOT / "artifacts"
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

SR = 22050
WINDOW_SEC = 30.0

#subprocess helpers
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(err or f"Command failed: {' '.join(cmd)}")
    return proc

#genre validation
def load_allowed_genres() -> list[str]:
    lm = ART / "label_map.json"
    if lm.exists():
        d = json.loads(lm.read_text(encoding="utf-8"))
        # keep order by id if possible
        inv = {v: k for k, v in d.items()}
        return [inv[i] for i in range(len(inv))]
    # fallback to canonical GTZAN set
    return ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def ensure_existing_genre(genre: str) -> Path:
    allowed = load_allowed_genres()
    if genre not in allowed:
        raise SystemExit(
            f"Genre '{genre}' is not in existing set: {allowed}\n"
            "Refusing to create a new genre."
        )
    gdir = DATA_ROOT / genre
    if not gdir.exists():
        raise SystemExit(
            f"Destination folder does not exist: {gdir}\n"
            "Refusing to create a new genre directory."
        )
    return gdir

#yt-dlp + ffmpeg
def yt_download(url: str, cookies_from: str | None, force_ipv4: bool) -> Path:
    base = TMP / f"yt_{uuid.uuid4().hex}"
    outtmpl = str(base) + ".%(ext)s"
    strategies = ["web", "android", "tv_embedded", "ios"]
    last = None
    for client in strategies:
        try:
            cmd = [
                "yt-dlp",
                "-f", "bestaudio/best",
                "-o", outtmpl,
                "--no-playlist",
                "--no-warnings",
                "--quiet",
                "--retries", "2",
                "--geo-bypass",
                "--extractor-args", f"youtube:player_client={client}",
            ]
            if cookies_from and cookies_from.lower() != "none":
                cmd += ["--cookies-from-browser", cookies_from]
            if force_ipv4:
                cmd += ["-4"]
            run(cmd + [url])
            files = list(TMP.glob(base.name + ".*"))
            if not files:
                raise RuntimeError("yt-dlp produced no file.")
            return files[0]
        except Exception as e:
            last = e
            for f in TMP.glob(base.name + ".*"):
                try: f.unlink()
                except Exception: pass
            continue
    raise RuntimeError(f"All yt-dlp strategies failed. Last error:\n{last}")

def probe_duration_seconds(path: Path) -> float | None:
    try:
        p = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1",str(path)])
        s = p.stdout.strip()
        return float(s) if s else None
    except Exception:
        return None

def middle_start(dur: float | None, window: float) -> float:
    if dur is None or dur <= 0 or dur < window:
        return 0.0
    mid = dur / 2.0
    return max(0.0, min(mid, dur - window))

def convert_trim_to_wav(src: Path, start_sec: float, window_sec: float) -> Path:
    wav = TMP / f"{uuid.uuid4().hex}.wav"
    cmd = [
        "ffmpeg","-y","-v","error",
        "-i", str(src),
        "-ss", f"{start_sec:.3f}",
        "-t", f"{window_sec:.3f}",
        "-ac","1",
        "-ar", str(SR),
        "-sample_fmt","s16",
        str(wav)
    ]
    run(cmd)
    return wav

def probe_audio_format(path: Path) -> dict:
    # Query key fields to assert exact format.
    proc = run([
        "ffprobe","-v","error","-show_streams","-select_streams","a:0",
        "-of","json", str(path)
    ])
    import json as _json
    data = _json.loads(proc.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No audio stream found.")
    s = streams[0]
    return {
        "channels": int(s.get("channels", 0)),
        "sample_rate": int(s.get("sample_rate", 0)),
        "sample_fmt": str(s.get("sample_fmt", "")),
        "codec_name": str(s.get("codec_name", "")),
    }

#naming
def next_index_for_genre(genre_dir: Path, genre: str) -> int:
    pat = re.compile(rf"^{re.escape(genre)}\.(\d+)\.wav$", re.IGNORECASE)
    max_idx = -1
    for p in genre_dir.glob("*.wav"):
        m = pat.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx + 1

def main():
    ap = argparse.ArgumentParser(description="Download YouTube → middle 30s → data/gtzan/<genre>/<genre>.<NNNNN>.wav")
    ap.add_argument("url", help="YouTube URL (quote it in PowerShell if it contains &)")
    ap.add_argument("genre", help="Target existing genre (must already exist; no new genres allowed)")
    ap.add_argument("--use-browser-cookies", default="none", choices=["none","chrome","edge","firefox"],
                    help="Use your browser cookies if the video is gated.")
    ap.add_argument("--ipv4", action="store_true", help="Force IPv4 for yt-dlp.")
    args = ap.parse_args()

    #Validate genre 
    genre_dir = ensure_existing_genre(args.genre)

    # 2)Download audio
    src = yt_download(args.url, cookies_from=args.use_browser_cookies, force_ipv4=args.ipv4)

    try:
        # 3)Trim to middle 30s and convert to mono@22050 s16
        dur = probe_duration_seconds(src)
        start = middle_start(dur, WINDOW_SEC)
        wav_tmp = convert_trim_to_wav(src, start, WINDOW_SEC)

        # 4)Verify
        meta = probe_audio_format(wav_tmp)
        if not (meta["channels"] == 1 and meta["sample_rate"] == SR and meta["sample_fmt"] == "s16"):
            # clean up if not matching strict contract
            try: wav_tmp.unlink()
            except Exception: pass
            raise SystemExit(
                f"Post-conversion format mismatch: {meta}. "
                "Expected channels=1, sample_rate=22050, sample_fmt='s16'."
            )

        # 5)Choose next filename
        idx = next_index_for_genre(genre_dir, args.genre)
        out = genre_dir / f"{args.genre}.{idx:05d}.wav"
        try:
            wav_tmp.replace(out)
        except Exception:
            out.write_bytes(wav_tmp.read_bytes())
            try: wav_tmp.unlink()
            except Exception: pass

        print(f"Saved: {out}")
        if dur is not None:
            print(f"Source duration ≈ {dur:.1f}s | Window: start={start:.1f}s, len={WINDOW_SEC:.0f}s")
        print("\nNext steps:")
        print("  - Rebuild metadata:   python .\\scripts\\make_metadata_gtzan.py")
        print("  - Re-split/extract/scale/train to include this clip.")

    finally:
        # remove original download
        try: src.unlink()
        except Exception: pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
