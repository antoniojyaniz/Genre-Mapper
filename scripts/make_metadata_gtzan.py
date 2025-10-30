from pathlib import Path
import csv
from collections import Counter

ROOT = Path("data/gtzan")
GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
OUT = Path("metadata.csv")

def main() -> None:
    rows = []
    for g in GENRES:
        gdir = ROOT / g
        if not gdir.exists():
            print(f"[warn] missing genre folder: {gdir}")
            continue
        for p in sorted(gdir.glob("*.wav")):
            rows.append([str(p).replace("\\","/"), g])

    if not rows:
        raise SystemExit("No files found under data/gtzan/<genre>/*.wav")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath","genre"])
        w.writerows(rows)

    cnt = Counter([g for _, g in rows])
    print(f"Wrote {OUT} with {len(rows)} rows.")
    for g in GENRES:
        print(f"{g:10s} : {cnt.get(g,0)}")

if __name__ == "__main__":
    main()
