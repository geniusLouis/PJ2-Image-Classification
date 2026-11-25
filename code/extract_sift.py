"""Extract OpenCV SIFT descriptors for every chapter 1 manifest row.

Each image is decoded, converted to grayscale and passed to SIFT with default
settings. All detected keypoints are kept; the 128-descriptor cap for BoVW is
applied in chapter 5. Images without keypoints contribute zero rows.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ("food101", "imagenette")
DIMENSION = 128
KEYPOINT_FIELDS = ["sample_id", "split", "label", "width", "height", "keypoints"]
SUMMARY_FIELDS = [
    "dataset", "images", "train", "test", "descriptor_dimension", "descriptors",
    "zero_keypoint_images", "zero_keypoint_fraction", "keypoints_mean", "keypoints_median",
    "keypoints_p95", "keypoints_max", "decode_seconds", "sift_seconds", "loop_seconds",
    "milliseconds_per_image", "cache_mib", "failures", "limit_per_split_label",
]


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {path}")
    if {row["split"] for row in rows} != {"train", "test"}:
        raise ValueError(f"Unexpected splits in {path}")
    return rows


def select_rows(rows, limit):
    """Keep manifest order. For smoke tests, cap rows per (split, label) group."""
    if limit is None:
        return rows
    seen = Counter()
    kept = []
    for row in rows:
        key = (row["split"], row["label"])
        if seen[key] < limit:
            seen[key] += 1
            kept.append(row)
    return kept


def decode_image(path, sample_id):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode sample {sample_id}: {path}")
    return image


def detect_descriptors(sift, gray, sample_id):
    """Return keypoints and a (K, 128) float32 array; K may be zero."""
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        # OpenCV returns None when no keypoints are found.
        return keypoints, np.empty((0, DIMENSION), np.float32)
    if descriptors.shape != (len(keypoints), DIMENSION) or descriptors.dtype != np.float32:
        raise ValueError(f"Unexpected SIFT output for sample {sample_id}: {descriptors.shape}")
    if not np.isfinite(descriptors).all():
        raise ValueError(f"Non-finite SIFT descriptor for sample {sample_id}")
    return keypoints, descriptors


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_examples(output, dataset, examples):
    count = len(examples)
    columns = min(count, 5)
    rows_needed = math.ceil(count / columns)
    fig, axes = plt.subplots(rows_needed, columns,
                             figsize=(2.7 * columns, 2.8 * rows_needed), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    example_rows = []
    for ax, (label, (sample_id, image, keypoints)) in zip(axes.ravel(), examples.items()):
        drawn = cv2.drawKeypoints(image, keypoints, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\n{len(keypoints)} keypoints", fontsize=8)
        example_rows.append(dict(label=label, sample_id=sample_id, split="train",
                                 keypoints=len(keypoints)))
    fig.tight_layout()
    fig.savefig(output / f"{dataset}_examples.png", dpi=150)
    plt.close(fig)
    (output / f"{dataset}_examples.json").write_text(
        json.dumps(example_rows, indent=2) + "\n", encoding="utf-8")


def extract_dataset(dataset, manifest, limit, output, cache_dir):
    rows = select_rows(read_manifest(manifest), limit)
    raw = ROOT / "data" / "raw"
    sift = cv2.SIFT_create()
    counts = np.zeros(len(rows), np.int64)
    blocks = []
    keypoint_rows = []
    examples = {}
    decode_seconds = 0.0
    sift_seconds = 0.0
    started = perf_counter()
    for i, row in enumerate(rows):
        tick = perf_counter()
        image = decode_image(raw / row["path"], row["sample_id"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        decode_seconds += perf_counter() - tick

        tick = perf_counter()
        keypoints, descriptors = detect_descriptors(sift, gray, row["sample_id"])
        sift_seconds += perf_counter() - tick

        counts[i] = len(keypoints)
        blocks.append(descriptors)
        height, width = image.shape[:2]
        keypoint_rows.append(dict(sample_id=row["sample_id"], split=row["split"],
                                  label=row["label"], width=width, height=height,
                                  keypoints=len(keypoints)))
        if row["split"] == "train" and row["label"] not in examples:
            examples[row["label"]] = (row["sample_id"], image, keypoints)
        if (i + 1) % 2000 == 0:
            print(f"{dataset}: {i + 1}/{len(rows)}", flush=True)
    loop_seconds = perf_counter() - started

    descriptors_all = np.concatenate(blocks) if blocks else np.empty((0, DIMENSION), np.float32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if offsets[-1] != len(descriptors_all):
        raise ValueError(f"Offsets do not match descriptor rows for {dataset}")

    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    splits = np.array([row["split"] for row in rows])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    # Publish the cache with a single rename, as in the chapter 2 script.
    temporary = cache_dir / f"{dataset}.tmp.npz"
    # Descriptors are stored uncompressed: compression is slow and saves little here.
    np.savez(temporary, descriptors=descriptors_all, offsets=offsets, sample_id=ids,
             label=labels, split=splits, manifest_sha256=np.array(manifest_sha256))
    temporary.replace(cache_dir / f"{dataset}.npz")

    write_csv(output / f"{dataset}_keypoints.csv", KEYPOINT_FIELDS, keypoint_rows)
    zeros = counts == 0
    images = len(rows)
    summary = dict(
        dataset=dataset, images=images,
        train=int((splits == "train").sum()), test=int((splits == "test").sum()),
        descriptor_dimension=DIMENSION, descriptors=int(counts.sum()),
        zero_keypoint_images=int(zeros.sum()), zero_keypoint_fraction=float(zeros.mean()),
        keypoints_mean=float(counts.mean()), keypoints_median=float(np.median(counts)),
        keypoints_p95=float(np.percentile(counts, 95)), keypoints_max=int(counts.max()),
        decode_seconds=decode_seconds, sift_seconds=sift_seconds, loop_seconds=loop_seconds,
        milliseconds_per_image=1000 * (decode_seconds + sift_seconds) / images,
        cache_mib=descriptors_all.nbytes / 1024**2, failures=0,
        limit_per_split_label="" if limit is None else limit,
    )
    plot_examples(output, dataset, examples)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test only: keep at most N rows per split and class.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sift")
    parser.add_argument("--cache", type=Path, default=ROOT / "results" / "cache" / "sift")
    args = parser.parse_args()

    cv2.setNumThreads(1)
    summaries = []
    for dataset in DATASETS:
        manifest = ROOT / "results" / "data_audit" / f"{dataset}_split.csv"
        summaries.append(extract_dataset(dataset, manifest, args.limit, args.output, args.cache))
        print(json.dumps(summaries[-1], indent=2), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", SUMMARY_FIELDS, summaries)
    sift = cv2.SIFT_create()
    metadata = dict(
        python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
        matplotlib=matplotlib.__version__, platform=platform.platform(),
        processor=platform.processor(), opencv_threads=1,
        sift=dict(nfeatures=sift.getNFeatures(), nOctaveLayers=sift.getNOctaveLayers(),
                  contrastThreshold=sift.getContrastThreshold(),
                  edgeThreshold=sift.getEdgeThreshold(), sigma=sift.getSigma()),
        color_space="grayscale via cv2.COLOR_BGR2GRAY", resize="none",
        descriptor="OpenCV SIFT, 128-dimensional float32, unnormalized, all keypoints kept",
        bovw_cap="128 descriptors per image applied in chapter 5, not here",
        randomness="none", seed=42,
        timing="Single pass; decode+grayscale and SIFT timed separately; excludes cache writing and plots",
        failure_policy="Raise with sample ID; zero-keypoint images keep zero rows",
        storage="descriptors float32 with offsets; uncompressed NPZ",
        limit_per_split_label=args.limit,
    )
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
