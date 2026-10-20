"""Quick CSV shape and label checks without loading full expression matrices."""
import argparse
import csv
from pathlib import Path

DATASETS = ("baronm1", "darmanis", "kolodziejczyk", "li", "pollen", "tasic", "xin", "zeisel")

def validate(data_dir):
    problems = []
    for name in DATASETS:
        matrix = data_dir / f"data_{name}.csv"
        labels = data_dir / f"labs_{name}.csv"
        if not matrix.exists() or not labels.exists():
            problems.append(f"{name}: missing matrix or labels")
            continue
        with matrix.open(newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh))
        with labels.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            if not ({"cell_type", "x"} & set(reader.fieldnames or [])):
                problems.append(f"{name}: missing cell_type/x label column")
        cells = len(header) - 1
        if cells != len(rows):
            problems.append(f"{name}: {cells} matrix cells vs {len(rows)} labels")
        print(f"{name}: {cells} cells, {len(rows)} labels")
    return problems

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    issues = validate(args.data_dir)
    if issues:
        raise SystemExit("\n".join(issues))
    print("All eight datasets passed structural checks.")
