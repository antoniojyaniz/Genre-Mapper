from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

ART = Path("artifacts")
SCALER_PATH = ART / "scaler.pkl"

def _load_X(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    if "X" not in d:
        raise SystemExit(f"{npz_path} missing 'X'")
    return d["X"]

def _save_X(npz_path: Path, X: np.ndarray) -> None:
    np.savez_compressed(npz_path, X=X.astype(np.float32))

def main() -> None:
    train_npz = ART / "train_features.npz"
    val_npz   = ART / "val_features.npz"
    test_npz  = ART / "test_features.npz"

    if not (train_npz.exists() and val_npz.exists() and test_npz.exists()):
        raise SystemExit("Missing feature files. Expected train/val/test .npz in artifacts/")

    Xtr = _load_X(train_npz)
    Xva = _load_X(val_npz)
    Xte = _load_X(test_npz)

    # Fit on Train only
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    _save_X(ART / "train_features_scaled.npz", Xtr_s)
    _save_X(ART / "val_features_scaled.npz",   Xva_s)
    _save_X(ART / "test_features_scaled.npz",  Xte_s)

    print(f"Saved scaler -> {SCALER_PATH}")
    print("Shapes (scaled):")
    print("  train:", Xtr_s.shape)
    print("  val  :", Xva_s.shape)
    print("  test :", Xte_s.shape)
    # sanity: check finite values
    if not np.isfinite(Xtr_s).all() or not np.isfinite(Xva_s).all() or not np.isfinite(Xte_s).all():
        raise SystemExit("Non-finite values after scaling. Investigate inputs.")

if __name__ == "__main__":
    main()
