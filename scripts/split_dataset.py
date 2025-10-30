import argparse
import os
from pathlib import Path
import pandas as pd

#optional sklearn stratification
try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None

ALLOWED = {"blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"}

def _counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("genre").size().sort_index()

def _manual_stratified_split(meta: pd.DataFrame, seed: int):
    # Per-genre shuffle, then 70/15/15 splits
    rng = pd.util.random.RandomState(seed)
    parts = []
    for g, gdf in meta.groupby("genre"):
        gdf = gdf.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        n = len(gdf); n_train = int(round(n*0.70)); n_val = int(round(n*0.15))
        train_g = gdf.iloc[:n_train]
        val_g   = gdf.iloc[n_train:n_train+n_val]
        test_g  = gdf.iloc[n_train+n_val:]
        parts.append(("train", train_g)); parts.append(("val", val_g)); parts.append(("test", test_g))
    train = pd.concat([df for name, df in parts if name=="train"], ignore_index=True)
    val   = pd.concat([df for name, df in parts if name=="val"], ignore_index=True)
    test  = pd.concat([df for name, df in parts if name=="test"], ignore_index=True)
    return train, val, test

def main(meta_path: str, out_dir: str, seed: int) -> None:
    print(f"CWD: {os.getcwd()}")
    meta = pd.read_csv(meta_path)
    expected = ["filepath","genre"]
    if list(meta.columns) != expected:
        raise SystemExit(f"Bad columns: {list(meta.columns)} != {expected}")
    bad = set(meta["genre"]) - ALLOWED
    if bad:
        raise SystemExit(f"Unknown genres: {sorted(bad)}")

    meta = meta.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

    if train_test_split is not None:
        train_df, temp_df = train_test_split(
            meta, test_size=0.30, random_state=seed, stratify=meta["genre"], shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=seed, stratify=temp_df["genre"], shuffle=True
        )
    else:
        print("[warn] sklearn not available; using manual stratified split")
        train_df, val_df, test_df = _manual_stratified_split(meta, seed)

    # ensure out dir exists (absolute)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    #write files
    train_fp = out_path / "train.csv"
    val_fp   = out_path / "val.csv"
    test_fp  = out_path / "test.csv"
    train_df.to_csv(train_fp, index=False)
    val_df.to_csv(val_fp, index=False)
    test_df.to_csv(test_fp, index=False)

    #sanity
    def overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
        return len(set(a["filepath"]).intersection(set(b["filepath"])))
    o1, o2, o3 = overlap(train_df, val_df), overlap(train_df, test_df), overlap(val_df, test_df)

    print(f"Rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("\n[Train]\n", _counts(train_df))
    print("\n[Val]\n", _counts(val_df))
    print("\n[Test]\n", _counts(test_df))
    print(f"\nOverlap: train∩val={o1}, train∩test={o2}, val∩test={o3}")
    print(f"\nWrote:\n  {train_fp}\n  {val_fp}\n  {test_fp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.meta, args.out, args.seed)