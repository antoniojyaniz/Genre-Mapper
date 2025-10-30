from pathlib import Path
import pandas as pd
import sys

ALLOWED = set(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"])

def main(path="metadata.csv"):
    df = pd.read_csv(path)
    expected_cols = ["filepath","genre"]
    if list(df.columns) != expected_cols:
        raise SystemExit(f"Bad columns: {list(df.columns)} != {expected_cols}")

    #genre whitelist
    bad = set(df["genre"]) - ALLOWED
    if bad:
        raise SystemExit(f"Unknown genres in CSV: {sorted(bad)}")

    #Files exist
    missing = [p for p in df["filepath"] if not Path(p).exists()]
    if missing:
        print(f"[error] missing files: {len(missing)} (showing first 10)")
        for m in missing[:10]:
            print(" -", m)
        raise SystemExit(1)

    print("OK:", len(df), "rows")
    print(df.groupby("genre").size())
    return 0

if __name__ == "__main__":
    sys.exit(main())