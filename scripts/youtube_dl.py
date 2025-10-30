import argparse
import json
import uuid
import subprocess
from pathlib import Path
import warnings

import joblib
import numpy as np
import librosa

warnings.filterwarnings("ignore", category=DeprecationWarning)

ART = Path("artifacts")
TMP = Path("tmp"); TMP.mkdir(exist_ok=True)

SR = 22050
N_MFCC = 20
WINDOW_SEC = 30.0  

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(err or f"Command failed: {' '.join(cmd)}")
    return proc

def _probe_duration_seconds(path: Path) -> float | None:
    try:
        proc = _run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1",
            str(path),
        ])
        s = proc.stdout.strip()
        return float(s) if s else None
    except Exception:
        return None

def _compute_middle_start(duration: float | None, window: float = WINDOW_SEC) -> float:
    if duration is None or duration <= 0 or duration < window:
        return 0.0
    mid = duration / 2.0
    start = max(0.0, min(mid, duration - window))
    return float(start)

def _yt_download_to_wav(url: str, cookies_from: str | None, force_ipv4: bool) -> Path:
    base = TMP / f"yt_{uuid.uuid4().hex}"
    outtmpl = str(base) + ".%(ext)s"

    strategies = ["web", "android", "tv_embedded", "ios"]
    last_error = None

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

            _run(cmd + [url])

            # Locate the downloaded source file
            candidates = list(TMP.glob(base.name + ".*"))
            if not candidates:
                raise RuntimeError("yt-dlp did not produce a file.")
            src = candidates[0]

            # Probe duration and compute middle-start
            dur = _probe_duration_seconds(src)
            start_sec = _compute_middle_start(dur, WINDOW_SEC)

            # Convert/trim to mono PCM16 WAV @ 22.05 kHz, 30s after the middle
            wav = TMP / f"{uuid.uuid4().hex}.wav"
            ff = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(src),
                "-ss", f"{start_sec:.3f}",
                "-t", f"{WINDOW_SEC:.3f}",
                "-ac", "1",
                "-ar", str(SR),
                "-sample_fmt", "s16",
                str(wav),
            ]
            _run(ff)

            try: src.unlink()
            except Exception: pass

            return wav
        except Exception as e:
            last_error = e
            # Clean up partials
            for f in TMP.glob(base.name + ".*"):
                try: f.unlink()
                except Exception: pass
            continue

    raise RuntimeError(
        "All yt-dlp strategies failed.\n"
        f"Last error:\n{last_error}\n\n"
        "Try:\n"
        "  • --use-browser-cookies chrome|edge|firefox\n"
        "  • --ipv4\n"
        "  • pip install -U yt-dlp\n"
        "  • a different video (some are gated/DRM)"
    )

def _agg_mean_std(x: np.ndarray):
    mu = np.mean(x, axis=1); sd = np.std(x, axis=1, ddof=0)
    return mu, sd

def _compute_features(y: np.ndarray, sr: int):
    feats, names = [], []

    #MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mu, sd = _agg_mean_std(mfcc)
    for i in range(N_MFCC):
        feats.append(float(mu[i])); names.append(f"mfcc_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"mfcc_{i+1}_std")

    #Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mu, sd = _agg_mean_std(chroma)
    for i in range(12):
        feats.append(float(mu[i])); names.append(f"chroma_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"chroma_{i+1}_std")

    #Spectral basics
    for arr, base in [
        (librosa.feature.spectral_centroid(y=y, sr=sr), "spec_centroid"),
        (librosa.feature.spectral_bandwidth(y=y, sr=sr), "spec_bandwidth"),
        (librosa.feature.spectral_rolloff(y=y, sr=sr), "spec_rolloff"),
    ]:
        mu, sd = _agg_mean_std(arr)
        feats.append(float(mu[0])); names.append(f"{base}_mean")
        feats.append(float(sd[0])); names.append(f"{base}_std")

    #Spectral contrast 
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mu, sd = _agg_mean_std(contrast)
    for i in range(contrast.shape[0]):
        feats.append(float(mu[i])); names.append(f"spec_contrast_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"spec_contrast_{i+1}_std")

    #RMS
    rms = librosa.feature.rms(y=y)
    mu, sd = _agg_mean_std(rms)
    feats.append(float(mu[0])); names.append("rms_mean")
    feats.append(float(sd[0])); names.append("rms_std")

    #ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    mu, sd = _agg_mean_std(zcr)
    feats.append(float(mu[0])); names.append("zcr_mean")
    feats.append(float(sd[0])); names.append("zcr_std")

    #Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_scalar = float(np.atleast_1d(tempo).ravel()[0])
    feats.append(tempo_scalar); names.append("tempo_bpm")

    return np.asarray(feats, dtype=np.float32), names

def _rationale(feature_map: dict[str, float], top1: str) -> str:
    tempo = feature_map.get("tempo_bpm", 0.0)
    bright = feature_map.get("spec_centroid_mean", 0.0)
    zcr = feature_map.get("zcr_mean", 0.0)
    if tempo > 120 and bright > 2500:
        return "High tempo + bright spectrum → dance/EDM-like energy."
    if tempo < 80 and bright < 1800 and zcr < 0.05:
        return "Low tempo + dark spectrum → acoustic/folk/classical traits."
    if bright > 3000 and zcr > 0.08:
        return "Very bright and edgy → rock/metal characteristics."
    return f"Feature mix (tempo≈{tempo:.0f} BPM, brightness≈{bright:.0f}) aligns with {top1}."

def main():
    p = argparse.ArgumentParser()
    p.add_argument("url", help="YouTube URL (quote it if it contains &)")
    p.add_argument("--use-browser-cookies", default="none",
                   choices=["none","chrome","edge","firefox"],
                   help="Use your signed-in browser cookies if the video is gated.")
    p.add_argument("--ipv4", action="store_true", help="Force IPv4 requests for yt-dlp.")
    args = p.parse_args()

    wav = _yt_download_to_wav(
        args.url,
        cookies_from=args.use_browser_cookies,
        force_ipv4=args.ipv4,
    )

    try:
        y, _ = librosa.load(str(wav), sr=SR, mono=True)

        vec, names = _compute_features(y, SR)

        order = json.loads((ART/"feature_order.json").read_text(encoding="utf-8"))
        if names != order:
            raise SystemExit("Feature order mismatch — training/inference schemas differ.")
        feature_map = {n: float(v) for n, v in zip(order, vec.tolist())}

        scaler = joblib.load(ART/"scaler.pkl")
        model  = joblib.load(ART/"model.pkl")
        X = scaler.transform(vec.reshape(1, -1))
        proba = model.predict_proba(X)[0]

        label_map = json.loads((ART/"label_map.json").read_text(encoding="utf-8"))
        inv = {v: k for k, v in label_map.items()}
        classes = [inv[i] for i in range(len(inv))]

        order_idx = np.argsort(proba)[::-1]
        top3 = [(classes[i], float(proba[i])) for i in order_idx[:3]]
        top1 = top3[0]

        why = _rationale(feature_map, top1[0])

        # 9) Print result
        print("\n== Prediction ==")
        print(f"Top-1: {top1[0]}  ({top1[1]*100:.1f}%)")
        print("Top-3:")
        for lbl, p in top3:
            print(f"  - {lbl:10s}  {p*100:5.1f}%")
        print(f"Reason: {why}")
        if (dur := _probe_duration_seconds(wav)) is not None:
            # This prints the trimmed file's duration (~30s), informative only
            print(f"Analyzed window: middle 30s (trimmed wav ≈ {dur:.1f}s)")
        else:
            print("Analyzed window: middle 30s")

    finally:
        try: wav.unlink()
        except Exception: pass

if __name__ == "__main__":
    main()
