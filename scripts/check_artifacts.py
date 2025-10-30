from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ART = Path("artifacts")

def load_npz(path): 
    d = np.load(path); assert "X" in d; return d["X"]

def main():
    must = [
        ART/"feature_order.json",
        ART/"label_map.json",
        ART/"scaler.pkl",
        ART/"model.pkl",
        ART/"train_features_scaled.npz",
        ART/"val_features_scaled.npz",
        ART/"test_features_scaled.npz",
        ART/"train_labels.csv",
        ART/"val_labels.csv",
        ART/"test_labels.csv",
    ]
    missing = [str(p) for p in must if not p.exists()]
    if missing:
        raise SystemExit("Missing artifacts:\n" + "\n".join(missing))

    # load
    order = json.loads((ART/"feature_order.json").read_text())
    label_map = json.loads((ART/"label_map.json").read_text())
    inv = {v:k for k,v in label_map.items()}
    scaler = joblib.load(ART/"scaler.pkl")
    model  = joblib.load(ART/"model.pkl")

    Xva = load_npz(ART/"val_features_scaled.npz")
    yva = pd.read_csv(ART/"val_labels.csv")["y"].to_numpy()

    # quick predictions
    proba = model.predict_proba(Xva[:5])
    pred  = proba.argmax(axis=1)
    print("Feature cols:", len(order))
    print("Classes:", [inv[i] for i in range(len(inv))])
    for i, (yi, pi) in enumerate(zip(yva[:5], pred)):
        print(f"val[{i}]: true={inv[yi]}  pred={inv[pi]}  conf={proba[i,pi]:.2f}")

    print("\nOK: artifacts load and predict successfully.")

if __name__ == "__main__":
    main()