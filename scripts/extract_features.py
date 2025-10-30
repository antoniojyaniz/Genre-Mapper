import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import uuid
import warnings
import numpy as np
import pandas as pd
import librosa
import subprocess
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

SR = 22050
N_MFCC = 20

ARTIFACTS = Path("artifacts")
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(exist_ok=True)
FEATURE_ORDER_PATH = ARTIFACTS / "feature_order.json"
LABEL_MAP_PATH = ARTIFACTS / "label_map.json"

# ---------- audio IO ----------
def _run_ffmpeg_convert(src: str, dst: str, sr: int = SR) -> None:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", src,
        "-ac", "1",
        "-ar", str(sr),
        "-sample_fmt", "s16",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed on {src}: {err}")

def _load_audio_any(path: str, sr: int = SR) -> np.ndarray:
    # Try direct load
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if y is None or y.size == 0 or np.all(~np.isfinite(y)):
            raise ValueError("empty/invalid audio after direct load")
        return y
    except Exception:
        # Fallback: transcode to clean PCM WAV then load
        tmp_wav = TMP_DIR / f"{uuid.uuid4().hex}.wav"
        try:
            _run_ffmpeg_convert(path, str(tmp_wav), sr=sr)
            y, _ = librosa.load(str(tmp_wav), sr=sr, mono=True)
            if y is None or y.size == 0 or np.all(~np.isfinite(y)):
                raise ValueError("empty/invalid audio after ffmpeg convert")
            return y
        finally:
            try:
                if tmp_wav.exists():
                    tmp_wav.unlink()
            except Exception:
                pass

#features
def _agg_mean_std(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=1)
    sd = np.std(x, axis=1, ddof=0)
    return mu, sd

def compute_feature_vector(path: str) -> Tuple[np.ndarray, List[str]]:
    y = _load_audio_any(path, sr=SR)

    feats: List[float] = []
    names: List[str] = []

    #MFCC(20) mean/std (40)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mu, sd = _agg_mean_std(mfcc)
    for i in range(N_MFCC):
        feats.append(float(mu[i])); names.append(f"mfcc_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"mfcc_{i+1}_std")

    #Chroma mean/std (24)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR)
    mu, sd = _agg_mean_std(chroma)
    for i in range(12):
        feats.append(float(mu[i])); names.append(f"chroma_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"chroma_{i+1}_std")

    #Spectral simple (6)
    for arr, base in [
        (librosa.feature.spectral_centroid(y=y, sr=SR), "spec_centroid"),
        (librosa.feature.spectral_bandwidth(y=y, sr=SR), "spec_bandwidth"),
        (librosa.feature.spectral_rolloff(y=y, sr=SR), "spec_rolloff"),
    ]:
        mu, sd = _agg_mean_std(arr)
        feats.append(float(mu[0])); names.append(f"{base}_mean")
        feats.append(float(sd[0])); names.append(f"{base}_std")

    #Spectral contrast (7 bands) mean/std (14)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR)
    mu, sd = _agg_mean_std(contrast)
    for i in range(contrast.shape[0]):
        feats.append(float(mu[i])); names.append(f"spec_contrast_{i+1}_mean")
        feats.append(float(sd[i])); names.append(f"spec_contrast_{i+1}_std")

    #RMS mean/std (2)
    rms = librosa.feature.rms(y=y)
    mu, sd = _agg_mean_std(rms)
    feats.append(float(mu[0])); names.append("rms_mean")
    feats.append(float(sd[0])); names.append("rms_std")

    # ZCR mean/std (2)
    zcr = librosa.feature.zero_crossing_rate(y)
    mu, sd = _agg_mean_std(zcr)
    feats.append(float(mu[0])); names.append("zcr_mean")
    feats.append(float(sd[0])); names.append("zcr_std")

    #Tempo BPM (1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    tempo_scalar = float(np.atleast_1d(tempo).ravel()[0])
    feats.append(tempo_scalar); names.append("tempo_bpm")

    vec = np.asarray(feats, dtype=np.float32)
    return vec, names

#helpers
def _save_feature_order(names: List[str]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FEATURE_ORDER_PATH.write_text(json.dumps(names, indent=2), encoding="utf-8")

def _assert_feature_order(names: List[str]) -> None:
    order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    if names != order:
        raise SystemExit("Feature order mismatch. Ensure extractor hasn't changed.")

def _load_or_create_label_map(genres: List[str]) -> Dict[str, int]:
    if LABEL_MAP_PATH.exists():
        return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    uniq = sorted(set(genres))
    mapping = {g: i for i, g in enumerate(uniq)}
    LABEL_MAP_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping

#driver
def extract_split(csv_in: Path, out_prefix: Path) -> None:
    df = pd.read_csv(csv_in)  # filepath,genre
    label_map = _load_or_create_label_map(df["genre"].tolist())

    X_list, y_list, names_ref = [], [], None
    failures: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        p = row["filepath"]; g = row["genre"]
        try:
            x, names = compute_feature_vector(p)
            if names_ref is None:
                names_ref = names
                if not FEATURE_ORDER_PATH.exists():
                    _save_feature_order(names_ref)
                else:
                    _assert_feature_order(names_ref)
            X_list.append(x)
            y_list.append(label_map[g])
        except Exception as e:
            failures.append((p, str(e)))
            continue  # skip bad file

    if not X_list:
        raise SystemExit(f"No features extracted from {csv_in}; all files failed?")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    np.savez_compressed(f"{out_prefix}_features.npz", X=X)
    pd.DataFrame({"y": y}).to_csv(f"{out_prefix}_labels.csv", index=False)

    # write failure log (if any)
    split_name = out_prefix.name  # train/val/test prefix expected
    if failures:
        log_path = ARTIFACTS / f"failed_decode_{split_name}.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for p, msg in failures:
                f.write(f"{p}\t{msg}\n")
        print(f"[warn] {len(failures)} files skipped. Logged to {log_path}")
        print("       First few:")
        for p, msg in failures[:5]:
            print("       -", p, "->", msg)

    print(f"Wrote {out_prefix}_features.npz with X.shape={X.shape}")
    print(f"Wrote {out_prefix}_labels.csv with {len(y)} labels")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="splits/train.csv or val.csv or test.csv")
    ap.add_argument("--out", required=True, help="artifacts/train (prefix) etc.")
    args = ap.parse_args()
    extract_split(Path(args.csv), Path(args.out))

if __name__ == "__main__":
    main()
