"""Compare global color and local BoVW predictions on matching test images."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def read_rows(path, test_only=False):
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if test_only and row["split"] != "test":
                continue
            key = row["sample_id"]
            if key in rows:
                raise ValueError(f"Duplicate sample ID in {path}: {key}")
            rows[key] = row
    return rows


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(manifest, global_rows, local_rows):
    if not manifest or manifest.keys() != global_rows.keys() or manifest.keys() != local_rows.keys():
        raise ValueError("Both prediction sets must match the official test IDs exactly.")
    labels = sorted({row["label"] for row in manifest.values()})
    matrices = {"global": Counter(), "local": Counter()}
    cases = []
    for key, row in manifest.items():
        truth = row["label"]
        color, local = global_rows[key], local_rows[key]
        if color["y_true"] != truth or local["y_true"] != truth:
            raise ValueError(f"Conflicting truth for {key}")
        a, b = color["y_pred"], local["y_pred"]
        if a not in labels or b not in labels:
            raise ValueError(f"Unknown predicted label for {key}")
        matrices["global"][truth, a] += 1
        matrices["local"][truth, b] += 1
        outcome = ("both_correct" if a == b == truth else "global_only" if a == truth
                   else "local_only" if b == truth else "both_wrong")
        cases.append(dict(sample_id=key, path=row["path"], truth=truth, global_pred=a,
                          local_pred=b, outcome=outcome))
    metrics, by_class = {}, []
    for method, matrix in matrices.items():
        f1s = []
        for label in labels:
            tp = matrix[label, label]
            actual = sum(matrix[label, other] for other in labels)
            predicted = sum(matrix[other, label] for other in labels)
            f1 = 2 * tp / (actual + predicted)
            f1s.append(f1)
            by_class.append(dict(method=method, label=label, support=actual,
                                 recall=tp / actual, f1=f1))
        metrics[method] = dict(accuracy=sum(matrix[c, c] for c in labels) / len(cases),
                               macro_f1=sum(f1s) / len(labels))
    outcomes = Counter(row["outcome"] for row in cases)
    return dict(n=len(cases), metrics=metrics,
                outcomes={k: outcomes[k] for k in
                          ("both_correct", "global_only", "local_only", "both_wrong")}), by_class, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.results
    output = args.output or base / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("food101", "imagenette"):
        inputs = [read_rows(base / "data_audit" / f"{name}_split.csv", test_only=True)]
        for folder in ("color_classification", "bovw_classification"):
            inputs.append(read_rows(base / folder / f"{name}_predictions.csv"))
        result, classes, cases = compare(*inputs)
        summary[name] = result
        write_csv(output / f"{name}_per_class.csv", classes)
        write_csv(output / f"{name}_paired_predictions.csv", cases)
        confusions = Counter((r["truth"], r[field], method) for r in cases
                             for method, field in (("global", "global_pred"), ("local", "local_pred"))
                             if r[field] != r["truth"])
        write_csv(output / f"{name}_confusions.csv",
                  [dict(truth=t, predicted=p, method=m, count=n)
                   for (t, p, m), n in confusions.most_common()],
                  ["truth", "predicted", "method", "count"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
