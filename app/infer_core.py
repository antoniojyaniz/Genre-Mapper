# app/infer_core.py
from __future__ import annotations
import json
import uuid
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import joblib
import librosa

ROOT = Path(".").resolve()
ART = ROOT / "artifacts"
TMP = ROOT / "tmp"; TMP.mkdir(exist_ok=True)
SR = 22050
N_MFCC = 20
WINDOW_SEC = 30.0

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(err or f"Command failed: {' '.join(cmd)}")
    return proc

def _probe_duration_seconds(path: Path) -> float | None:
    try:
        p = _run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1",str(path)])
        s = p.stdout.strip()
        return float(s) if s else None
    except Exception:
        return None

def _compute_middle_start(dur: float | None, window: float = WINDOW_SEC) -> float:
    if dur is None or dur <= 0 or dur < window:
        return 0.0
    mid = dur / 2.0
    return max(0.0, min(mid, dur - window))

def _yt_download_middle_wav(url: str, cookies_from: str = "none", force_ipv4: bool = False) -> Tuple[Path, Dict[str, float]]:
    base = TMP / f"yt_{uuid.uuid4().hex}"
    outtmpl = str(base) + ".%(ext)s"
    strategies = ["web", "android", "tv_embedded", "ios"]
    last_error = None
    for client in strategies:
        try:
            cmd = [
                "yt-dlp","-f","bestaudio/best","-o",outtmpl,
                "--no-playlist","--no-warnings","--quiet","--retries","2","--geo-bypass",
                "--extractor-args", f"youtube:player_client={client}",
            ]
            if cookies_from and cookies_from.lower() != "none":
                cmd += ["--cookies-from-browser", cookies_from]
            if force_ipv4:
                cmd += ["-4"]
            _run(cmd + [url])

            srcs = list(TMP.glob(base.name + ".*"))
            if not srcs:
                raise RuntimeError("yt-dlp produced no file.")
            src = srcs[0]

            dur = _probe_duration_seconds(src)
            start_sec = _compute_middle_start(dur, WINDOW_SEC)

            wav = TMP / f"{uuid.uuid4().hex}.wav"
            ff = [
                "ffmpeg","-y","-v","error",
                "-i",str(src),
                "-ss",f"{start_sec:.3f}",
                "-t",f"{WINDOW_SEC:.3f}",
                "-ac","1","-ar",str(SR),"-sample_fmt","s16",
                str(wav)
            ]
            _run(ff)
            try: src.unlink()
            except Exception: pass

            return wav, {"duration": float(dur) if dur else None, "start_sec": start_sec, "window_sec": WINDOW_SEC}
        except Exception as e:
            last_error = e
            for f in TMP.glob(base.name + ".*"):
                try: f.unlink()
                except Exception: pass
            continue
    raise RuntimeError(
        "All yt-dlp strategies failed.\n"
        f"Last error:\n{last_error}\n\n"
        "Try signed-in cookies: --use-browser-cookies chrome | edge | firefox, or force IPv4."
    )

def _agg_mean_std(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(x, axis=1), np.std(x, axis=1, ddof=0)

def _compute_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    feats: List[float] = []
    names: List[str] = []

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mu, sd = _agg_mean_std(mfcc)
    for i in range(N_MFCC):
        feats.append(float(mu[i])); names.append(f"mfcc_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"mfcc_{i+1}_std")

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mu, sd = _agg_mean_std(chroma)
    for i in range(12):
        feats.append(float(mu[i])); names.append(f"chroma_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"chroma_{i+1}_std")

    # Spectral centroid/bandwidth/rolloff
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mu_c, sd_c = _agg_mean_std(spec_centroid)
    feats.append(float(mu_c[0])); names.append("spec_centroid_mean")
    feats.append(float(sd_c[0])); names.append("spec_centroid_std")

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mu_b, sd_b = _agg_mean_std(spec_bandwidth)
    feats.append(float(mu_b[0])); names.append("spec_bandwidth_mean")
    feats.append(float(sd_b[0])); names.append("spec_bandwidth_std")

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mu_r, sd_r = _agg_mean_std(spec_rolloff)
    feats.append(float(mu_r[0])); names.append("spec_rolloff_mean")
    feats.append(float(sd_r[0])); names.append("spec_rolloff_std")

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mu, sd = _agg_mean_std(contrast)
    for i in range(contrast.shape[0]):
        feats.append(float(mu[i])); names.append(f"spec_contrast_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"spec_contrast_{i+1}_std")

    # RMS
    rms = librosa.feature.rms(y=y)
    mu_rms, sd_rms = _agg_mean_std(rms)
    feats.append(float(mu_rms[0])); names.append("rms_mean")
    feats.append(float(sd_rms[0])); names.append("rms_std")

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    mu_z, sd_z = _agg_mean_std(zcr)
    feats.append(float(mu_z[0])); names.append("zcr_mean")
    feats.append(float(sd_z[0])); names.append("zcr_std")

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_scalar = float(np.atleast_1d(tempo).ravel()[0])
    feats.append(tempo_scalar); names.append("tempo_bpm")

    vec = np.asarray(feats, dtype=np.float32)

    # UI stats
    stats = {
        "tempo_bpm": tempo_scalar,
        "brightness_hz": float(spec_centroid.mean()),
        "energy_rms": float(rms.mean()),
        "noisiness_zcr": float(zcr.mean()),
        "bandwidth_hz": float(spec_bandwidth.mean()),
        "rolloff_hz": float(spec_rolloff.mean()),
    }
    return vec, names, stats

def analyze_youtube_url(url: str, cookies_from: str = "none", force_ipv4: bool = False) -> Dict[str, Any]:
    wav, _win = _yt_download_middle_wav(url, cookies_from=cookies_from, force_ipv4=force_ipv4)
    try:
        y, _ = librosa.load(str(wav), sr=SR, mono=True)
        vec, names, stats = _compute_features(y, SR)

        order = json.loads((ART/"feature_order.json").read_text(encoding="utf-8"))
        if names != order:
            raise RuntimeError("Feature order mismatch; retrain/inference schema differ.")

        scaler = joblib.load(ART/"scaler.pkl")
        model  = joblib.load(ART/"model.pkl")
        X = scaler.transform(vec.reshape(1, -1))
        proba = model.predict_proba(X)[0]

        label_map = json.loads((ART/"label_map.json").read_text(encoding="utf-8"))
        inv = {v: k for k, v in label_map.items()}
        classes = [inv[i] for i in range(len(inv))]

        order_idx = np.argsort(proba)[::-1]
        top3 = [{"label": classes[i], "conf": float(proba[i])} for i in order_idx[:3]]
        top1 = top3[0]

        return {
            "top1": top1,
            "top3": top3,
            "stats": stats,
            "model_version": (ART/"model_version.txt").read_text(encoding="utf-8").strip()
                              if (ART/"model_version.txt").exists() else "unknown",
        }
    finally:
        try: wav.unlink()
        except Exception: pass
