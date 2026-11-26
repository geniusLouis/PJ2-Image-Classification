"""Train and evaluate the chapter 5 SIFT/BoVW classifier."""

import argparse
import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import local
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw"
DATASETS = {"food101": "Food-101", "imagenette": "Imagenette"}
SAMPLE_COUNTS = {"food101": (4500, 1500), "imagenette": (9469, 3925)}
SIFT_CAP = 128
KS = (64, 128, 256)
NORMALIZATIONS = {"l1": "L1", "l2": "L2", "tfidf_l2": "TF-IDF + L2", "raw": "Raw count"}
BASELINE_K = 128
BASELINE_NORMALIZATION = "l1"
DICTIONARY_SAMPLE = 100_000
KMEANS_SETTINGS = dict(batch_size=4096, n_init=3, max_iter=100, max_no_improvement=10)
FIGURE_FONTS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
}


def descriptor_indices(count, sample_id, seed):
    if count <= SIFT_CAP:
        return slice(None)
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(count, SIFT_CAP, replace=False))


def extract_capped(rows, dataset, seed, workers):
    """Rebuild just the BoVW input using chapter 4's detector and settings."""
    from extract_sift import DIMENSION, decode_image, detect_descriptors

    thread_state = local()

    def extract(row):
        sift = getattr(thread_state, "sift", None)
        if sift is None:
            sift = thread_state.sift = cv2.SIFT_create()
        started = perf_counter()
        image = decode_image(RAW / row["path"], row["sample_id"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        decoded = perf_counter()
        keypoints, descriptors = detect_descriptors(sift, gray, row["sample_id"])
        detected = perf_counter()
        kept = descriptors[descriptor_indices(len(descriptors), row["sample_id"], seed)]
        return len(keypoints), kept, decoded - started, detected - decoded, perf_counter() - detected

    counts = np.zeros(len(rows), dtype=np.int32)
    dense = np.zeros((len(rows), SIFT_CAP, DIMENSION), dtype=np.float32)
    times = {"train": np.zeros(3), "test": np.zeros(3)}
    cv2.setNumThreads(1)
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            for i, (count, kept, decode, sift, cap) in enumerate(pool.map(extract, batch), start):
                counts[i] = count
                dense[i, :len(kept)] = kept
                times[rows[i]["split"]] += (decode, sift, cap)
            if (start // 256 + 1) % 8 == 0 or start + len(batch) == len(rows):
                print(f"{dataset}: SIFT {start + len(batch)}/{len(rows)}", flush=True)
    metadata = dict(
        source="extracted locally with chapter 4 detector", opencv=cv2.__version__,
        sift_workers=workers, opencv_threads_per_worker=1,
        sift_extraction_wall_seconds=perf_counter() - started,
    )
    for split, durations in times.items():
        for stage, seconds in zip(("decode", "sift", "cap"), durations, strict=True):
            metadata[f"{split}_{stage}_worker_seconds"] = float(seconds)
    return dense, counts, metadata


def read_chapter4(cache_path, rows, manifest_hash, seed):
    """Cap the full SIFT archive in bounded batches without loading it into RAM."""
    from extract_sift import DIMENSION

    with np.load(cache_path, allow_pickle=False) as cache:
        if cache["manifest_sha256"].item() != manifest_hash:
            raise ValueError(f"Stale chapter 4 manifest in {cache_path}.")
        for field in ("sample_id", "label", "split"):
            if not np.array_equal(cache[field], [row[field] for row in rows]):
                raise ValueError(f"Mismatched {field} in {cache_path}.")
        offsets = cache["offsets"]
    if offsets.shape != (len(rows) + 1,) or offsets.dtype != np.int64 or offsets[0] != 0:
        raise ValueError(f"Invalid SIFT offsets in {cache_path}.")
    counts = np.diff(offsets)
    if (counts < 0).any():
        raise ValueError(f"Out-of-order SIFT offsets in {cache_path}.")
    dense = np.zeros((len(rows), SIFT_CAP, DIMENSION), dtype=np.float32)
    started = perf_counter()
    with ZipFile(cache_path) as archive, archive.open("descriptors.npy") as stream:
        version = np.lib.format.read_magic(stream)
        read_header = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        if read_header is None:
            raise ValueError(f"Unsupported NPY version {version} in {cache_path}.")
        shape, fortran_order, dtype = read_header(stream)
        if shape != (offsets[-1], DIMENSION) or dtype != np.float32 or fortran_order:
            raise ValueError(f"Invalid descriptor storage in {cache_path}.")
        for start in range(0, len(rows), 64):
            end = min(start + 64, len(rows))
            count = offsets[end] - offsets[start]
            data = stream.read(int(count * DIMENSION * dtype.itemsize))
            blocks = np.frombuffer(data, dtype=dtype).reshape(count, DIMENSION)
            boundaries = offsets[start:end + 1] - offsets[start]
            for j in range(start, end):
                low, high = boundaries[j - start:j - start + 2]
                kept = blocks[low:high][descriptor_indices(int(counts[j]), rows[j]["sample_id"], seed)]
                dense[j, :len(kept)] = kept
    return dense, counts, dict(
        source="chapter 4 descriptor cache", cache_cap_seconds=perf_counter() - started,
    )


def load_descriptors(dataset, rows, manifest_hash, seed, workers, extract_missing):
    cache_dir = RESULTS / "cache"
    chapter4 = cache_dir / "sift" / f"{dataset}.npz"
    compact = cache_dir / "bovw" / f"{dataset}_descriptors.npz"
    fingerprint = f"1:{manifest_hash}:{seed}:{SIFT_CAP}:{cv2.__version__}"
    if compact.is_file():
        with np.load(compact, allow_pickle=False) as cache:
            if cache["fingerprint"].item() == fingerprint:
                return cache["descriptors"], cache["counts"], json.loads(cache["metadata"].item())
    if chapter4.is_file():
        dense, counts, metadata = read_chapter4(chapter4, rows, manifest_hash, seed)
    elif extract_missing:
        dense, counts, metadata = extract_capped(rows, dataset, seed, workers)
    else:
        raise FileNotFoundError(
            f"Missing {chapter4}. Run code/extract_sift.py or pass --extract-missing."
        )
    if not np.isfinite(dense).all():
        raise ValueError(f"Invalid capped SIFT descriptors for {dataset}.")
    compact.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=compact.parent) as temporary:
        staged = Path(temporary) / compact.name
        np.savez(staged, descriptors=dense, counts=counts,
                 fingerprint=fingerprint, metadata=json.dumps(metadata))
        staged.replace(compact)
    return dense, counts, metadata


def dataset_manifest(dataset, read_csv):
    from extract_sift import read_manifest

    base = RESULTS / "data_audit"
    manifest = base / f"{dataset}_split.csv"
    rows = read_manifest(manifest)
    label_rows = read_csv(base / f"{dataset}_labels.csv")
    labels = [row["label"] for row in sorted(label_rows, key=lambda row: int(row["label_id"]))]
    y = np.array([row["label"] for row in rows])
    train = np.flatnonzero([row["split"] == "train" for row in rows])
    test = np.flatnonzero([row["split"] == "test" for row in rows])
    if (
        (len(train), len(test)) != SAMPLE_COUNTS[dataset]
        or set(y[train]) != set(labels)
        or set(y[test]) != set(labels)
    ):
        raise ValueError(f"Unexpected official split sizes or classes in {manifest}.")
    return rows, labels, y, train, test, hashlib.sha256(manifest.read_bytes()).hexdigest()


def sample_training_descriptors(descriptors, counts, training, seed):
    """Sample uniformly over capped descriptors belonging to the fitting images."""
    offsets = np.r_[0, np.cumsum(counts[training])]
    size = min(DICTIONARY_SAMPLE, int(offsets[-1]))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(int(offsets[-1]), size, replace=False)
    images = np.searchsorted(offsets, sampled, side="right") - 1
    return descriptors[training[images], sampled - offsets[images]]


def encode_histograms(dictionary, descriptors, counts, images):
    """Assign each retained descriptor to its nearest center and count by image."""
    k = dictionary.n_clusters
    histograms = np.zeros((len(images), k), dtype=np.float32)
    positions = np.arange(SIFT_CAP)
    for start in range(0, len(images), 128):
        batch = images[start:start + 128]
        present = positions < counts[batch, None]
        if present.any():
            owners = np.nonzero(present)[0]
            words = dictionary.predict(descriptors[batch][present])
            histograms[start:start + len(batch)] = np.bincount(
                owners * k + words, minlength=len(batch) * k,
            ).reshape(len(batch), k)
    if not np.array_equal(histograms.sum(axis=1), counts[images]):
        raise ValueError("Quantization counts do not match retained descriptors.")
    return histograms


def fit_idf(histograms, method):
    if method == "tfidf_l2":
        document_frequency = np.count_nonzero(histograms, axis=0)
        return (np.log((1 + len(histograms)) / (1 + document_frequency)) + 1).astype(np.float32)
    return None


def transform_histograms(histograms, method, idf):
    from sklearn.preprocessing import normalize

    if method == "raw":
        return histograms
    weighted = histograms if idf is None else histograms * idf
    return normalize(weighted, norm="l1" if method == "l1" else "l2")


def timed(function, *args):
    start = perf_counter()
    value = function(*args)
    return value, perf_counter() - start


def fit_dictionary(sample, k, seed):
    from sklearn.cluster import MiniBatchKMeans

    if len(sample) < k:
        raise ValueError(f"Only {len(sample)} training descriptors for K={k}.")
    return MiniBatchKMeans(n_clusters=k, random_state=seed, **KMEANS_SETTINGS).fit(sample)


def search_key(row):
    return (-row["mean_macro_f1"], row["K"], row["C"], tuple(NORMALIZATIONS).index(row["normalization"]))


def search_parameters(dataset, descriptors, counts, y, train, labels):
    from classify_color import C_VALUES, SEED, fit_classifier, training_folds
    from evaluate import score_predictions

    trials = []
    dictionary_trials = []
    assignments = np.empty(len(train), dtype=int)
    for fold, (fit_idx, val_idx) in enumerate(training_folds(y[train]), 1):
        assignments[val_idx] = fold
        fit, val = train[fit_idx], train[val_idx]
        sample, sample_seconds = timed(sample_training_descriptors, descriptors, counts, fit, SEED + fold)
        for k in KS:
            dictionary, dictionary_seconds = timed(fit_dictionary, sample, k, SEED + fold)
            raw_fit, fit_quantization = timed(encode_histograms, dictionary, descriptors, counts, fit)
            raw_val, val_quantization = timed(encode_histograms, dictionary, descriptors, counts, val)
            dictionary_trials.append(dict(
                dataset=dataset, fold=fold, K=k, fit_samples=len(fit), validation_samples=len(val),
                dictionary_descriptors=len(sample), sample_seconds=sample_seconds,
                dictionary_seconds=dictionary_seconds, fit_quantization_seconds=fit_quantization,
                validation_quantization_seconds=val_quantization,
            ))
            for method in NORMALIZATIONS:
                idf, idf_seconds = timed(fit_idf, raw_fit, method)
                x_fit, fit_transform = timed(transform_histograms, raw_fit, method, idf)
                x_val, val_transform = timed(transform_histograms, raw_val, method, idf)
                for c in C_VALUES:
                    model, fit_seconds = timed(fit_classifier, x_fit, y[fit], c)
                    predicted, predict_seconds = timed(model.predict, x_val)
                    scores = score_predictions(y[val], predicted, labels)
                    trials.append(dict(
                        dataset=dataset, fold=fold, K=k, normalization=method, C=c,
                        accuracy=scores["accuracy"], macro_f1=scores["macro_f1"],
                        idf_seconds=idf_seconds, fit_transform_seconds=fit_transform,
                        validation_transform_seconds=val_transform,
                        fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                        iterations=int(model.n_iter_),
                    ))
            print(f"{dataset}: completed fold {fold}, K={k}", flush=True)

    grouped = {}
    for row in trials:
        grouped.setdefault((row["K"], row["normalization"], row["C"]), []).append(row)
    search = []
    for (k, method, c), candidate in grouped.items():
        f1 = [row["macro_f1"] for row in candidate]
        accuracy = [row["accuracy"] for row in candidate]
        search.append(dict(
            dataset=dataset, K=k, normalization=method, C=c,
            mean_macro_f1=float(np.mean(f1)), std_macro_f1=float(np.std(f1)),
            mean_accuracy=float(np.mean(accuracy)), std_accuracy=float(np.std(accuracy)),
            mean_svc_fit_seconds=float(np.mean([row["fit_seconds"] for row in candidate])),
            mean_svc_predict_seconds=float(np.mean([row["predict_seconds"] for row in candidate])),
        ))
    return sorted(search, key=search_key), trials, dictionary_trials, assignments


def plot_k_tradeoff(path, searches, costs):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4.1), layout="constrained")
    for ax, (dataset, name) in zip(fig.subplots(1, 2), DATASETS.items(), strict=True):
        cv = [min((r for r in searches if r["dataset"] == dataset and r["K"] == k
                   and r["normalization"] == BASELINE_NORMALIZATION), key=search_key) for k in KS]
        ax.errorbar(KS, [100 * r["mean_macro_f1"] for r in cv],
                    yerr=[100 * r["std_macro_f1"] for r in cv],
                    marker="o", capsize=4, color="#266184", label="Macro-F1")
        axes2 = ax.twinx()
        for field, label, color in (
            ("dictionary_seconds", "Dictionary fit", "#b46128"),
            ("fit_quantization_seconds", "Fit + validation encoding", "#52733c"),
        ):
            means = []
            for k in KS:
                fold_costs = [r for r in costs if r["dataset"] == dataset and r["K"] == k]
                times = [r[field] + (r["validation_quantization_seconds"]
                                    if field == "fit_quantization_seconds" else 0)
                         for r in fold_costs]
                means.append(np.mean(times))
            axes2.plot(KS, means, "--s", color=color, label=label)
        ax.set(title=name, xlabel="Vocabulary size K", ylabel="Validation Macro-F1 (%)", xticks=KS)
        axes2.set_ylabel("Mean seconds per fold")
        ax.grid(axis="y", alpha=0.25)
        lines, row_labels = ax.get_legend_handles_labels()
        other_lines, other_labels = axes2.get_legend_handles_labels()
    fig.legend(lines + other_lines, row_labels + other_labels,
               loc="outside lower center", ncols=3, fontsize=9)
    fig.savefig(path, dpi=200)


def plot_normalization(path, searches):
    from matplotlib.figure import Figure

    methods = ("raw", "l1", "l2", "tfidf_l2")
    fig = Figure(figsize=(9.2, 4), layout="constrained")
    for ax, (dataset, name) in zip(fig.subplots(1, 2), DATASETS.items(), strict=True):
        best = [min((r for r in searches if r["K"] == BASELINE_K and r["dataset"] == dataset
                     and r["normalization"] == method), key=search_key) for method in methods]
        x = np.arange(len(methods))
        means = [100 * r["mean_macro_f1"] for r in best]
        errors = [100 * r["std_macro_f1"] for r in best]
        ax.errorbar(x, means, yerr=errors, fmt="o", color="#266184", capsize=5)
        for i, (row, mean, error) in enumerate(zip(best, means, errors, strict=True)):
            ax.annotate(f'{mean:.1f}%\nC={row["C"]:g}', (i, mean + error), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)
        low, high = min(np.array(means) - errors), max(np.array(means) + errors)
        ax.set_ylim(low - 1, high + max((high - low) * 0.3, 2))
        ax.set(xlim=(-0.45, 3.45), title=name, ylabel="Validation Macro-F1 (%)",
               xticks=x, xticklabels=[NORMALIZATIONS[method] for method in methods])
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=200)


def plot_confusions(path, matrices):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 5.8), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, (dataset, (labels, counts)) in zip(axes, matrices.items(), strict=True):
        counts = np.asarray(counts)
        proportions = counts / counts.sum(axis=1, keepdims=True)
        plot = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
        ticks = np.arange(len(labels))
        readable = [label.replace("_", " ") for label in labels]
        ax.set(title=DATASETS[dataset], xlabel="Predicted class", ylabel="True class",
               xticks=ticks, xticklabels=readable, yticks=ticks, yticklabels=readable)
        ax.tick_params(axis="both", labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(55)
            tick.set_horizontalalignment("right")
        for (i, j), count in np.ndenumerate(counts):
            ax.text(j, i, str(count), ha="center", va="center", fontsize=7,
                    color="white" if proportions[i, j] > 0.55 else "#162943")
    fig.colorbar(plot, ax=axes, shrink=0.68, label="Fraction within true class")
    fig.savefig(path, dpi=230)


def prepare_example(row, dictionary, raw_histogram, method, idf, expected_count):
    from classify_color import SEED
    from extract_sift import decode_image, detect_descriptors

    sample_id = row["sample_id"]
    image = decode_image(RAW / row["path"], sample_id)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detect_descriptors(cv2.SIFT_create(), gray, sample_id)
    kept = np.arange(len(keypoints))[descriptor_indices(len(keypoints), sample_id, SEED)]
    # A chapter 4 cache may come from another OpenCV build. Use the same local
    # detections for the overlay and the example histogram.
    visual_counts = np.zeros(dictionary.n_clusters, dtype=np.float32)
    if len(kept):
        visual_counts[:] = np.bincount(dictionary.predict(descriptors[kept]),
                                       minlength=dictionary.n_clusters)
    overlay = cv2.drawKeypoints(image, [keypoints[i] for i in kept], None, color=(51, 221, 253),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    display_method = BASELINE_NORMALIZATION if method == "raw" else method
    hist = transform_histograms(visual_counts[None], display_method, idf)[0]
    details = dict(sample_id=sample_id, label=row["label"], keypoints=len(keypoints),
                   retained=len(kept), normalization=display_method,
                   nonzero_bins=int(np.count_nonzero(hist)),
                   matches_cached_encoding=(len(keypoints) == expected_count
                                            and np.array_equal(visual_counts, raw_histogram)))
    return image, overlay, hist, details


def plot_examples(path, examples):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10.5, 5.8), layout="constrained")
    axes = fig.subplots(2, 3, gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for row, (dataset, (image, overlay, hist, example)) in zip(axes, examples.items(), strict=True):
        for ax, pixels, title in (
            (row[0], image, f'{DATASETS[dataset]} · {example["label"].replace("_", " ")}'),
            (row[1], overlay, f'SIFT keypoints used ({example["retained"]})'),
        ):
            ax.imshow(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        row[2].bar(np.arange(1, len(hist) + 1), hist, color="#266184", width=0.9)
        row[2].set(title=f'{NORMALIZATIONS[example["normalization"]]} BoVW histogram',
                   xlabel="Visual word index", ylabel="Weight", xlim=(0, len(hist) + 1))
        row[2].grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=230)


def publish_dataset(output, dataset, rows, labels, y, test, predictions, scores, train, assignments):
    from prepare_splits import write_csv

    write_csv(output / f"{dataset}_predictions.csv", ["sample_id", "y_true", "y_pred"], [
        dict(sample_id=rows[i]["sample_id"], y_true=y[i], y_pred=predicted)
        for i, predicted in zip(test, predictions, strict=True)
    ])
    write_csv(output / f"{dataset}_per_class.csv", list(scores["per_class"][0]), scores["per_class"])
    write_csv(output / f"{dataset}_confusion.csv", ["y_true", *labels], [
        dict(y_true=label, **dict(zip(labels, counts, strict=True)))
        for label, counts in zip(labels, scores["confusion_matrix"], strict=True)
    ])
    write_csv(output / f"{dataset}_folds.csv", ["sample_id", "label", "validation_fold"], [
        dict(sample_id=rows[i]["sample_id"], label=y[i], validation_fold=int(fold))
        for i, fold in zip(train, assignments, strict=True)
    ])


def run_dataset(dataset, output, options):
    from classify_color import SEED, fit_classifier, plot_confusion, read_csv
    from evaluate import score_predictions

    rows, labels, y, train, test, manifest_hash = dataset_manifest(dataset, read_csv)
    (descriptors, original_counts, sift_timing), cache_seconds = timed(
        load_descriptors, dataset, rows, manifest_hash, SEED, options.sift_workers, options.extract_missing,
    )
    counts = np.minimum(original_counts, SIFT_CAP)
    cv_started = perf_counter()
    search, trials, k_trials, assignments = search_parameters(dataset, descriptors, counts, y, train, labels)
    cv_seconds = perf_counter() - cv_started
    best = search[0]
    k, method, c = best["K"], best["normalization"], best["C"]
    sample, sample_seconds = timed(sample_training_descriptors, descriptors, counts, train, SEED)
    dictionary, dictionary_seconds = timed(fit_dictionary, sample, k, SEED)
    raw_train, train_quantization = timed(encode_histograms, dictionary, descriptors, counts, train)
    idf, idf_seconds = timed(fit_idf, raw_train, method)
    x_train, train_transform = timed(transform_histograms, raw_train, method, idf)
    model, fit_seconds = timed(fit_classifier, x_train, y[train], c)
    raw_test, test_quantization = timed(encode_histograms, dictionary, descriptors, counts, test)
    x_test, test_transform = timed(transform_histograms, raw_test, method, idf)
    predictions, predict_seconds = timed(model.predict, x_test)
    scores = score_predictions(y[test], predictions, labels)

    publish_dataset(output, dataset, rows, labels, y, test, predictions, scores, train, assignments)
    plot_confusion(output / f"{dataset}_confusion.png", scores["confusion_matrix"], labels, DATASETS[dataset])
    example_idx = int(np.flatnonzero(counts[train])[0])
    i = train[example_idx]
    example = prepare_example(rows[i], dictionary, raw_train[example_idx], method, idf, original_counts[i])
    baseline = min((row for row in search if row["K"] == BASELINE_K
                    and row["normalization"] == BASELINE_NORMALIZATION), key=search_key)

    summary = dict(
        dataset=dataset, K=k, dimension=k, normalization=method, C=c,
        train_samples=len(train), test_samples=len(test), dictionary_descriptors=len(sample),
        cv_accuracy=best["mean_accuracy"], cv_macro_f1=best["mean_macro_f1"],
        cv_std_macro_f1=best["std_macro_f1"], accuracy=scores["accuracy"], macro_f1=scores["macro_f1"],
        sift_extraction_wall_seconds=sift_timing.get("sift_extraction_wall_seconds", ""),
        cv_seconds=cv_seconds, sample_seconds=sample_seconds, dictionary_seconds=dictionary_seconds,
        train_quantization_seconds=train_quantization, test_quantization_seconds=test_quantization,
        idf_seconds=idf_seconds, train_transform_seconds=train_transform, test_transform_seconds=test_transform,
        fit_seconds=fit_seconds, predict_seconds=predict_seconds, iterations=int(model.n_iter_),
    )
    chapter4 = next(row for row in read_csv(RESULTS / "sift" / "summary.csv") if row["dataset"] == dataset)
    config = dict(
        **summary, manifest_sha256=manifest_hash, labels=labels, baseline=baseline, sift=sift_timing,
        chapter4_sift_reference_seconds=dict(decode=float(chapter4["decode_seconds"]),
                                             sift=float(chapter4["sift_seconds"])),
        capped_cache_preparation_or_load_seconds=cache_seconds,
        retained_descriptor_mean=float(counts.mean()), retained_descriptors=int(counts.sum()),
        retained_descriptor_min=int(counts.min()), images_capped=int((original_counts > SIFT_CAP).sum()),
        zero_descriptor_images={split: int((counts[indices] == 0).sum())
                                for split, indices in (("train", train), ("test", test))},
        example=example[-1],
    )
    print(f"{dataset}: selected K={k}, {method}, C={c:g}; "
          f"test Accuracy={scores['accuracy']:.6f}, Macro-F1={scores['macro_f1']:.6f}", flush=True)
    return summary, search, trials, k_trials, config, (labels, scores["confusion_matrix"]), example


def main():
    import matplotlib
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits

    from classify_color import C_VALUES, FOLDS, SEED, SVC_SETTINGS
    from prepare_splits import write_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-missing", action="store_true",
                        help="Rebuild capped descriptors from raw images if the chapter 4 cache is absent.")
    parser.add_argument("--sift-workers", type=int, default=1,
                        help="Parallel image workers used only when extracting missing descriptors.")
    options = parser.parse_args()
    if options.sift_workers < 1:
        parser.error("--sift-workers must be positive")

    matplotlib.use("Agg")
    cv2.setNumThreads(1)
    output = RESULTS / "bovw_classification"
    output.mkdir(parents=True, exist_ok=True)
    # Stage the full evaluation before replacing the chapter 6 handoff files.
    with (
        TemporaryDirectory(dir=output.parent) as temporary,
        threadpool_limits(limits=1),
        matplotlib.rc_context(FIGURE_FONTS),
    ):
        staged = Path(temporary)
        summaries, searches, fold_trials, dictionary_trials = [], [], [], []
        datasets, matrices, examples = {}, {}, {}
        for dataset in DATASETS:
            summary, search, trials, costs, config, matrix, example = run_dataset(dataset, staged, options)
            summaries.append(summary)
            searches.extend(search)
            fold_trials.extend(trials)
            dictionary_trials.extend(costs)
            datasets[dataset], matrices[dataset], examples[dataset] = config, matrix, example
        for file, records in (
            ("summary.csv", summaries), ("cv_search.csv", searches), ("cv_folds.csv", fold_trials),
            ("cv_dictionary_timing.csv", dictionary_trials),
        ):
            write_csv(staged / file, list(records[0]), records)
        plot_k_tradeoff(staged / "figure_5_1_k_tradeoff.png", searches, dictionary_trials)
        plot_normalization(staged / "figure_5_2_normalization.png", searches)
        plot_confusions(staged / "figure_5_3_confusions.png", matrices)
        plot_examples(staged / "figure_5_4_encoding.png", examples)

        config = dict(
            seed=SEED, folds=FOLDS, shuffle=True, K_candidates=KS, C_candidates=C_VALUES,
            normalizations=NORMALIZATIONS, baseline=dict(K=BASELINE_K, normalization=BASELINE_NORMALIZATION),
            selection="Largest mean validation Macro-F1, then smaller K, smaller C, then normalization order.",
            cv_dictionary_seed="seed + fold (1-based)", final_dictionary_seed=SEED,
            sift_cap_per_image=SIFT_CAP,
            descriptor_capping="Uniform without replacement, seeded by SHA-256 of seed and sample ID.",
            zero_keypoint_policy="Keep the image with an all-zero histogram for every normalization.",
            dictionary_sample_cap=DICTIONARY_SAMPLE,
            dictionary_sample_policy="Uniform without replacement from capped descriptors of fitting images only.",
            dictionary=dict(estimator="MiniBatchKMeans", **KMEANS_SETTINGS),
            quantization="Nearest Euclidean center, one visual word per retained descriptor.",
            tfidf="Raw count multiplied by ln((1+n_fit)/(1+df_fit))+1, then L2. IDF fits on fitting images only.",
            estimator=dict(name="LinearSVC", **SVC_SETTINGS),
            numerical_threads=1, test_policy="Evaluate only the selected final configuration on the official test set.",
            timing="Dictionary fit, quantization, normalization, classifier fit and predict timed separately, "
                   "excluding SIFT, file IO and scoring. SIFT worker seconds sum per-image intervals and may "
                   "exceed elapsed extraction wall time. Chapter 4 reference is a separate run.",
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            opencv=cv2.__version__, scikit_learn=sklearn.__version__, matplotlib=matplotlib.__version__,
            platform=platform.platform(), processor=platform.processor(), datasets=datasets,
        )
        (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for file in sorted(staged.iterdir()):
            file.replace(output / file.name)


if __name__ == "__main__":
    main()
