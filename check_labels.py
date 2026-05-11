import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/main")
for split in ["train", "valid", "test"]:
    ldir = DATA_DIR / split / "labels"
    if not ldir.exists(): continue
    for lf in ldir.glob("*.txt"):
        lines = [ln.strip() for ln in lf.read_text().split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) > 0:
                try:
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id >= 23:
                        print(f"File {lf} has invalid class ID {cls_id} on line {i+1}: {line}")
                except ValueError:
                    print(f"File {lf} has invalid format on line {i+1}: {line}")
print("Check done.")
