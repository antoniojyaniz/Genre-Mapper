from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

ART = Path("artifacts")
MODEL_PATH = ART / "model.pkl"
METRICS_PATH = ART / "metrics.txt"
VERSION_PATH = ART / "model_version.txt"
CONF_MAT_PATH = ART / "confusion_matrix.png"
LABEL_MAP_PATH = ART / "label_map.json"

def _load_X(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    if "X" not in d:
        raise SystemExit(f"{npz_path} missing 'X'")
    return d["X"]

def _load_y(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "y" not in df.columns:
        raise SystemExit(f"{csv_path} missing column 'y'")
    return df["y"].to_numpy(dtype=np.int64)

def main(seed: int = 42):
    #Load scaled features
    Xtr = _load_X(ART / "train_features_scaled.npz")
    Xva = _load_X(ART / "val_features_scaled.npz")
    Xte = _load_X(ART / "test_features_scaled.npz")

    #Load labels
    ytr = _load_y(ART / "train_labels.csv")
    yva = _load_y(ART / "val_labels.csv")
    yte = _load_y(ART / "test_labels.csv")

    #Class names (index → name) 
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    #label_map is {genre: id}; invert to list ordered by id
    inv = {v: k for k, v in label_map.items()}
    classes_names = [inv[i] for i in range(len(inv))]

    #Hyperparameter search over C using Val Macro-F1
    best = (-1.0, None, None)
    for C in [0.1, 1.0, 3.0, 10.0]:
        base = LinearSVC(C=C, random_state=seed)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        clf.fit(Xtr, ytr)
        yva_pred = clf.predict(Xva)
        f1m = f1_score(yva, yva_pred, average="macro")
        if f1m > best[0]:
            best = (f1m, C, clf)

    best_f1_val, best_C, best_clf = best

    #Final evaluation 
    yte_pred = best_clf.predict(Xte)
    acc = accuracy_score(yte, yte_pred)
    f1m_test = f1_score(yte, yte_pred, average="macro")

    #Save 
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODEL_PATH)

    #Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        yte, yte_pred, display_labels=classes_names, xticks_rotation=45, cmap=None
    )
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(CONF_MAT_PATH, dpi=160)
    plt.close()

    # Metrics
    METRICS_PATH.write_text(
        "\n".join([
            f"Best C (Val Macro-F1): {best_C}",
            f"Val Macro-F1 (best): {best_f1_val:.4f}",
            f"Test Accuracy: {acc:.4f}",
            f"Test Macro-F1: {f1m_test:.4f}",
        ]),
        encoding="utf-8"
    )
    VERSION_PATH.write_text("genre-v1.0\n", encoding="utf-8")

    print(f"Best C: {best_C}  |  Val Macro-F1: {best_f1_val:.4f}")
    print(f"Test Accuracy: {acc:.4f}  |  Test Macro-F1: {f1m_test:.4f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Wrote {CONF_MAT_PATH} and {METRICS_PATH}")
    print(f"Model version -> {VERSION_PATH.read_text(encoding='utf-8').strip()}")

if __name__ == "__main__":
    main()