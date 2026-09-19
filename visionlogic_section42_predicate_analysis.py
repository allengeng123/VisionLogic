"""Numerical analysis of VisionLogic pathway and predicate structure.

The input checkpoint contains initial prediction-preserving pathways without
ordered-signature extension. Class-conditioned thresholds use source extrema.
Validation pathways are extracted with the same minimum-sufficient-prefix
procedure, conditioned on each frozen model prediction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


MODEL_HEADS = {
    "vit": ("heads.head.weight", "heads.head.bias", "stable"),
    "resnet": ("fc.weight", "fc.bias", "stable"),
    "convnext": ("classifier.2.weight", "classifier.2.bias", None),
    "swin": ("head.weight", "head.bias", None),
}
CUTOFFS = (0.01, 0.05, 0.10, 0.25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def activation_files(path: Path) -> list[Path]:
    files = sorted(
        item for item in path.iterdir() if item.suffix.lower() in {".pkl", ".pickle"}
    )
    if len(files) != 1000:
        raise RuntimeError(f"Expected 1000 activation files in {path}, found {len(files)}")
    return files


def load_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise TypeError(f"Unexpected activation payload in {path}: {type(value)!r}")
    return value


def percentile(values, q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def describe(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("count", "mean", "std", "p05", "q1", "median", "q3", "p95")}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5)),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "q3": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def concentration(counts: np.ndarray) -> dict:
    positive = np.asarray(counts[counts > 0], dtype=np.int64)
    if positive.size == 0:
        return {"observed": 0, "occurrences": 0, "n50": 0, "n80": 0, "n90": 0}
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    return {
        "observed": int(positive.size),
        "occurrences": int(ordered.sum()),
        "n50": int(np.searchsorted(cumulative, 0.50) + 1),
        "n80": int(np.searchsorted(cumulative, 0.80) + 1),
        "n90": int(np.searchsorted(cumulative, 0.90) + 1),
    }


def curve(counts: np.ndarray) -> dict:
    positive = np.sort(np.asarray(counts[counts > 0], dtype=np.float64))[::-1]
    if positive.size == 0:
        return {"predicate_fraction": [0.0, 1.0], "occurrence_share": [0.0, 0.0]}
    return {
        "predicate_fraction": np.arange(1, positive.size + 1).astype(float) / positive.size,
        "occurrence_share": np.cumsum(positive) / positive.sum(),
    }


def physical_active_counts(active: np.ndarray, physical_ids: np.ndarray) -> np.ndarray:
    unique_physical, inverse = np.unique(physical_ids, return_inverse=True)
    result = np.zeros((active.shape[0], len(unique_physical)), dtype=bool)
    for column, target in enumerate(inverse):
        result[:, target] |= active[:, column]
    return np.sum(result, axis=1)


def active_matrix(
    matrix: np.ndarray,
    logical_ids: np.ndarray,
    physical_ids: np.ndarray,
    thresholds: np.ndarray,
    dimension: int,
) -> np.ndarray:
    values = matrix[:, physical_ids]
    positive = logical_ids < dimension
    return np.where(positive[None, :], values >= thresholds[None, :], values <= thresholds[None, :])


def add_image_values(store, group: str, **values) -> None:
    for name, array in values.items():
        store[group][name].append(np.asarray(array))


def finalize_image_values(store) -> dict:
    result = {}
    for group, metrics in store.items():
        result[group] = {}
        for name, chunks in metrics.items():
            combined = np.concatenate(chunks) if chunks else np.asarray([])
            result[group][name] = describe(combined)
    return result


def extract_path(row, predicted_class: int, weights, bias, dimension: int, sort_kind):
    contributions = row * weights[predicted_class]
    if sort_kind is None:
        ranked = np.argsort(contributions)[::-1]
    else:
        ranked = np.argsort(contributions, kind=sort_kind)[::-1]
    logical = ranked.copy()
    logical[row[ranked] < 0] += dimension
    partial = bias.copy()
    selected = []
    for position, physical_id in enumerate(ranked):
        selected.append(int(logical[position]))
        partial += weights[:, physical_id] * row[physical_id]
        if int(np.argmax(partial)) == predicted_class:
            break
    return selected


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    kwargs = {"mode": "wt", "newline": "", "encoding": "utf-8"}
    handle_context = gzip.open(path, **kwargs) if path.suffix == ".gz" else path.open(**kwargs)
    with handle_context as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_index(train_files, weights, bias):
    mapping = {}
    raw_rows = 0
    majority_rows_before_overwrite = 0
    for label, path in enumerate(train_files):
        matrix = load_matrix(path)
        predictions = np.argmax(matrix @ weights.T + bias, axis=1)
        values, counts = np.unique(predictions, return_counts=True)
        majority = int(values[np.argmax(counts)])
        selected_indices = np.flatnonzero(predictions == majority)
        mapping[majority] = {
            "path": path,
            "source_label": label,
            "indices": selected_indices,
            "majority_class": majority,
        }
        raw_rows += len(matrix)
        majority_rows_before_overwrite += len(selected_indices)
        if (label + 1) % 100 == 0:
            print(f"indexed train files {label + 1}/1000", flush=True)
    return mapping, raw_rows, majority_rows_before_overwrite


def class_record(split, class_id, denominator, vocab_size, activation_counts, selection_counts):
    activation_frequency = activation_counts / max(denominator, 1)
    selection_frequency = selection_counts / max(denominator, 1)
    act_con = concentration(activation_counts)
    sel_con = concentration(selection_counts)
    record = {
        "split": split,
        "class_id": int(class_id),
        "images": int(denominator),
        "vocabulary_size_signed": int(vocab_size),
        "activation_observed": act_con["observed"],
        "selection_observed": sel_con["observed"],
        "activation_occurrences": act_con["occurrences"],
        "selection_occurrences": sel_con["occurrences"],
        "activation_n50": act_con["n50"],
        "activation_n80": act_con["n80"],
        "activation_n90": act_con["n90"],
        "selection_n50": sel_con["n50"],
        "selection_n80": sel_con["n80"],
        "selection_n90": sel_con["n90"],
        "activation_n80_fraction": act_con["n80"] / max(act_con["observed"], 1),
        "selection_n80_fraction": sel_con["n80"] / max(sel_con["observed"], 1),
    }
    for cutoff in CUTOFFS:
        suffix = str(int(cutoff * 100))
        record[f"activation_ge_{suffix}pct"] = int(np.count_nonzero(activation_frequency >= cutoff))
        record[f"selection_ge_{suffix}pct"] = int(np.count_nonzero(selection_frequency >= cutoff))
    return record


def predicate_record(
    model,
    split,
    class_id,
    logical_id,
    dimension,
    threshold,
    denominator,
    activation_count,
    selection_count,
    joint_count,
    samples="",
):
    return {
        "model": model,
        "split": split,
        "class_id": int(class_id),
        "logical_id": int(logical_id),
        "physical_id": int(logical_id % dimension),
        "branch": "+" if logical_id < dimension else "-",
        "threshold": float(threshold),
        "images": int(denominator),
        "activation_count": int(activation_count),
        "selection_count": int(selection_count),
        "joint_count": int(joint_count),
        "activation_frequency": float(activation_count / max(denominator, 1)),
        "selection_frequency": float(selection_count / max(denominator, 1)),
        "joint_frequency": float(joint_count / max(denominator, 1)),
        "selection_given_active": float(joint_count / activation_count) if activation_count else None,
        "selected_satisfies_threshold": float(joint_count / selection_count) if selection_count else None,
        "sample_ids": samples,
    }


def analyze_model(args) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    weight_key, bias_key, sort_kind = MODEL_HEADS[args.model]
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    weights = state[weight_key].detach().cpu().numpy()
    bias = state[bias_key].detach().cpu().numpy()
    dimension = int(weights.shape[1])
    train_files = activation_files(args.train_dir)
    val_files = activation_files(args.val_dir)
    with args.phase3.open("rb") as handle:
        payload = pickle.load(handle)
    pathways = {int(key): value for key, value in payload["resolved_pruned_dataset"].items()}

    print(f"{args.model}: indexing training prediction populations", flush=True)
    sources, raw_train_rows, pre_overwrite_rows = source_index(train_files, weights, bias)
    if set(pathways) != set(sources):
        raise RuntimeError(
            f"Checkpoint/source classes differ: missing={sorted(set(pathways)-set(sources))[:10]}, "
            f"extra={sorted(set(sources)-set(pathways))[:10]}"
        )

    predicate_rows = []
    class_rows = []
    reuse = defaultdict(list)
    thresholds_by_class = {}
    train_vectors = {}
    image_values = defaultdict(lambda: defaultdict(list))
    global_counts = Counter()
    source_manifest = []

    print(f"{args.model}: fitting original class-specific thresholds and analyzing source pathways", flush=True)
    for number, class_id in enumerate(sorted(pathways), start=1):
        source = sources[class_id]
        full = load_matrix(source["path"])
        matrix = full[source["indices"]]
        class_paths = pathways[class_id]
        if len(matrix) != len(class_paths):
            raise RuntimeError(
                f"Class {class_id}: {len(matrix)} selected activations but {len(class_paths)} pathways"
            )
        source_manifest.append(
            {
                "class_id": class_id,
                "source_file": source["path"].name,
                "source_label": source["source_label"],
                "images": len(matrix),
            }
        )
        selection_counter = Counter()
        selected_indices_by_predicate = defaultdict(list)
        selected_signed = np.zeros(len(matrix), dtype=np.int32)
        selected_physical = np.zeros(len(matrix), dtype=np.int32)
        for image_index, path in enumerate(class_paths):
            logical_path = [int(value) for value in path]
            selection_counter.update(logical_path)
            for logical_id in logical_path:
                if len(selected_indices_by_predicate[logical_id]) < 3:
                    selected_indices_by_predicate[logical_id].append(image_index)
            selected_signed[image_index] = len(logical_path)
            selected_physical[image_index] = len({value % dimension for value in logical_path})

        logical_ids = np.asarray(sorted(selection_counter), dtype=np.int64)
        physical_ids = logical_ids % dimension
        lid_to_column = {int(value): index for index, value in enumerate(logical_ids)}
        threshold_values = np.where(
            logical_ids < dimension, np.inf, -np.inf
        ).astype(np.float32)
        for image_index, path in enumerate(class_paths):
            for value in path:
                logical_id = int(value)
                column = lid_to_column[logical_id]
                activation_value = matrix[image_index, physical_ids[column]]
                if logical_id < dimension:
                    threshold_values[column] = min(
                        threshold_values[column], activation_value
                    )
                else:
                    threshold_values[column] = max(
                        threshold_values[column], activation_value
                    )

        active = active_matrix(matrix, logical_ids, physical_ids, threshold_values, dimension)
        activation_counts = np.sum(active, axis=0).astype(np.int64)
        selection_counts = np.asarray([selection_counter[int(value)] for value in logical_ids], dtype=np.int64)
        joint_counts = np.zeros(len(logical_ids), dtype=np.int64)
        both_signed = np.zeros(len(matrix), dtype=np.int32)
        both_physical = np.zeros(len(matrix), dtype=np.int32)
        for image_index, path in enumerate(class_paths):
            active_selected = []
            for value in path:
                logical_id = int(value)
                column = lid_to_column[logical_id]
                if active[image_index, column]:
                    joint_counts[column] += 1
                    active_selected.append(logical_id)
            both_signed[image_index] = len(active_selected)
            both_physical[image_index] = len({value % dimension for value in active_selected})

        active_signed = np.sum(active, axis=1)
        active_physical = physical_active_counts(active, physical_ids)
        vocabulary_physical = len(np.unique(physical_ids))
        is_correct = class_id == source["source_label"]
        correctness = np.full(len(matrix), is_correct, dtype=bool)
        values = {
            "active_signed": active_signed,
            "active_physical": active_physical,
            "selected_signed": selected_signed,
            "selected_physical": selected_physical,
            "both_signed": both_signed,
            "both_physical": both_physical,
            "active_fraction_signed_vocabulary": active_signed / max(len(logical_ids), 1),
            "active_fraction_physical_dimension": active_physical / dimension,
            "selected_fraction_physical_dimension": selected_physical / dimension,
            "selected_threshold_fraction": both_signed / np.maximum(selected_signed, 1),
        }
        add_image_values(image_values, "train_all", **values)
        add_image_values(
            image_values,
            "train_correct" if is_correct else "train_incorrect",
            **values,
        )

        class_rows.append(
            {
                **class_record(
                    "train", class_id, len(matrix), len(logical_ids), activation_counts, selection_counts
                ),
                "vocabulary_size_physical": vocabulary_physical,
                "source_label": source["source_label"],
                "all_predictions_correct_for_source_label": bool(is_correct),
                "selected_threshold_joint_occurrences": int(joint_counts.sum()),
                "selected_occurrences_outside_vocabulary": 0,
            }
        )
        train_vectors[class_id] = {
            "activation": activation_counts,
            "selection": selection_counts,
        }
        thresholds_by_class[class_id] = {
            "logical_ids": logical_ids,
            "physical_ids": physical_ids,
            "thresholds": threshold_values,
            "lid_to_column": lid_to_column,
        }
        for column, logical_id in enumerate(logical_ids):
            sample_ids = ";".join(
                f"{source['path'].name}:{int(source['indices'][idx])}"
                for idx in selected_indices_by_predicate[int(logical_id)]
            )
            row = predicate_record(
                args.model,
                "train",
                class_id,
                int(logical_id),
                dimension,
                threshold_values[column],
                len(matrix),
                activation_counts[column],
                selection_counts[column],
                joint_counts[column],
                sample_ids,
            )
            predicate_rows.append(row)
            reuse[int(logical_id)].append(row)
        global_counts["train_images"] += len(matrix)
        global_counts["train_activation_occurrences"] += int(activation_counts.sum())
        global_counts["train_selection_occurrences"] += int(selection_counts.sum())
        global_counts["train_joint_occurrences"] += int(joint_counts.sum())
        if number % 100 == 0:
            print(f"{args.model}: analyzed source classes {number}/{len(pathways)}", flush=True)

    print(f"{args.model}: extracting held-out prediction-conditioned pathways", flush=True)
    val_groups = defaultdict(lambda: {"x": [], "labels": [], "paths": [], "sample_ids": []})
    raw_val_rows = 0
    val_correct = 0
    for label, path in enumerate(val_files):
        matrix = load_matrix(path)
        predictions = np.argmax(matrix @ weights.T + bias, axis=1)
        raw_val_rows += len(matrix)
        val_correct += int(np.count_nonzero(predictions == label))
        for row_index, (row, predicted_class) in enumerate(zip(matrix, predictions)):
            predicted_class = int(predicted_class)
            group = val_groups[predicted_class]
            group["x"].append(row)
            group["labels"].append(label)
            group["sample_ids"].append(f"{path.name}:{row_index}")
            group["paths"].append(
                extract_path(row, predicted_class, weights, bias, dimension, sort_kind)
            )
        if (label + 1) % 100 == 0:
            print(f"{args.model}: extracted validation files {label + 1}/1000", flush=True)

    val_missing_threshold_classes = []
    for class_number, class_id in enumerate(sorted(val_groups), start=1):
        group = val_groups[class_id]
        matrix = np.asarray(group["x"])
        labels = np.asarray(group["labels"], dtype=np.int64)
        correct_mask = labels == class_id
        class_paths = group["paths"]
        threshold_data = thresholds_by_class.get(class_id)
        if threshold_data is None:
            val_missing_threshold_classes.append(class_id)
            continue
        logical_ids = threshold_data["logical_ids"]
        physical_ids = threshold_data["physical_ids"]
        threshold_values = threshold_data["thresholds"]
        lid_to_column = threshold_data["lid_to_column"]
        active = active_matrix(matrix, logical_ids, physical_ids, threshold_values, dimension)
        activation_counts = np.sum(active, axis=0).astype(np.int64)
        selection_counts = np.zeros(len(logical_ids), dtype=np.int64)
        joint_counts = np.zeros(len(logical_ids), dtype=np.int64)
        selected_signed = np.zeros(len(matrix), dtype=np.int32)
        selected_physical = np.zeros(len(matrix), dtype=np.int32)
        both_signed = np.zeros(len(matrix), dtype=np.int32)
        both_physical = np.zeros(len(matrix), dtype=np.int32)
        outside_vocabulary = 0
        for image_index, path in enumerate(class_paths):
            selected_signed[image_index] = len(path)
            selected_physical[image_index] = len({value % dimension for value in path})
            active_selected = []
            for logical_id in path:
                column = lid_to_column.get(int(logical_id))
                if column is None:
                    outside_vocabulary += 1
                    continue
                selection_counts[column] += 1
                if active[image_index, column]:
                    joint_counts[column] += 1
                    active_selected.append(int(logical_id))
            both_signed[image_index] = len(active_selected)
            both_physical[image_index] = len({value % dimension for value in active_selected})

        active_signed = np.sum(active, axis=1)
        active_physical = physical_active_counts(active, physical_ids)
        values = {
            "active_signed": active_signed,
            "active_physical": active_physical,
            "selected_signed": selected_signed,
            "selected_physical": selected_physical,
            "both_signed": both_signed,
            "both_physical": both_physical,
            "active_fraction_signed_vocabulary": active_signed / max(len(logical_ids), 1),
            "active_fraction_physical_dimension": active_physical / dimension,
            "selected_fraction_physical_dimension": selected_physical / dimension,
            "selected_threshold_fraction": both_signed / np.maximum(selected_signed, 1),
        }
        add_image_values(image_values, "val_all", **values)
        if np.any(correct_mask):
            add_image_values(
                image_values,
                "val_correct",
                **{name: np.asarray(value)[correct_mask] for name, value in values.items()},
            )
        if np.any(~correct_mask):
            add_image_values(
                image_values,
                "val_incorrect",
                **{name: np.asarray(value)[~correct_mask] for name, value in values.items()},
            )

        class_rows.append(
            {
                **class_record(
                    "val", class_id, len(matrix), len(logical_ids), activation_counts, selection_counts
                ),
                "vocabulary_size_physical": len(np.unique(physical_ids)),
                "source_label": "",
                "all_predictions_correct_for_source_label": "",
                "selected_threshold_joint_occurrences": int(joint_counts.sum()),
                "selected_occurrences_outside_vocabulary": int(outside_vocabulary),
            }
        )
        for column, logical_id in enumerate(logical_ids):
            predicate_rows.append(
                predicate_record(
                    args.model,
                    "val",
                    class_id,
                    int(logical_id),
                    dimension,
                    threshold_values[column],
                    len(matrix),
                    activation_counts[column],
                    selection_counts[column],
                    joint_counts[column],
                )
            )
        global_counts["val_analyzed_images"] += len(matrix)
        global_counts["val_activation_occurrences"] += int(activation_counts.sum())
        global_counts["val_selection_eligible_occurrences"] += int(selection_counts.sum())
        global_counts["val_selection_outside_vocabulary"] += int(outside_vocabulary)
        global_counts["val_joint_occurrences"] += int(joint_counts.sum())
        global_counts["val_correct_analyzed_images"] += int(np.count_nonzero(correct_mask))
        global_counts["val_incorrect_analyzed_images"] += int(np.count_nonzero(~correct_mask))
        if class_number % 100 == 0:
            print(f"{args.model}: analyzed validation prediction classes {class_number}/{len(val_groups)}", flush=True)

    reuse_rows = []
    for logical_id, entries in sorted(reuse.items()):
        activation_frequencies = np.asarray([row["activation_frequency"] for row in entries])
        selection_frequencies = np.asarray([row["selection_frequency"] for row in entries])
        total_selection = selection_frequencies.sum()
        if total_selection > 0 and len(entries) > 1:
            proportions = selection_frequencies / total_selection
            entropy = -float(np.sum(proportions * np.log(proportions + 1e-15))) / math.log(len(entries))
            hhi = float(np.sum(proportions**2))
        else:
            entropy = 0.0
            hhi = 1.0
        top_entries = sorted(entries, key=lambda row: row["selection_frequency"], reverse=True)[:3]
        reuse_rows.append(
            {
                "model": args.model,
                "logical_id": logical_id,
                "physical_id": logical_id % dimension,
                "branch": "+" if logical_id < dimension else "-",
                "predicate_classes": len(entries),
                "classes_activation_ge_1pct": int(np.count_nonzero(activation_frequencies >= 0.01)),
                "classes_selection_ge_1pct": int(np.count_nonzero(selection_frequencies >= 0.01)),
                "classes_selection_ge_5pct": int(np.count_nonzero(selection_frequencies >= 0.05)),
                "mean_class_activation_frequency": float(np.mean(activation_frequencies)),
                "std_class_activation_frequency": float(np.std(activation_frequencies)),
                "mean_class_selection_frequency": float(np.mean(selection_frequencies)),
                "std_class_selection_frequency": float(np.std(selection_frequencies)),
                "selection_class_entropy": entropy,
                "selection_class_hhi": hhi,
                "representative_samples": "|".join(
                    f"class={row['class_id']}:{row['sample_ids']}" for row in top_entries
                ),
            }
        )

    train_class_rows = [row for row in class_rows if row["split"] == "train"]
    concentration_scores = np.asarray(
        [1.0 - row["selection_n80_fraction"] for row in train_class_rows]
    )
    representatives = {}
    for label, quantile in (("low", 10), ("median", 50), ("high", 90)):
        target = np.percentile(concentration_scores, quantile)
        index = int(np.argmin(np.abs(concentration_scores - target)))
        selected = train_class_rows[index]
        class_id = int(selected["class_id"])
        representatives[label] = {
            "class_id": class_id,
            "selection_concentration_score": float(concentration_scores[index]),
            "selection": curve(train_vectors[class_id]["selection"]),
            "activation": curve(train_vectors[class_id]["activation"]),
        }

    predicate_fields = [
        "model", "split", "class_id", "logical_id", "physical_id", "branch", "threshold",
        "images", "activation_count", "selection_count", "joint_count", "activation_frequency",
        "selection_frequency", "joint_frequency", "selection_given_active",
        "selected_satisfies_threshold", "sample_ids",
    ]
    class_fields = sorted({key for row in class_rows for key in row})
    reuse_fields = list(reuse_rows[0]) if reuse_rows else []
    write_rows(output / "predicate_metrics.csv.gz", predicate_fields, predicate_rows)
    write_rows(output / "class_metrics.csv", class_fields, class_rows)
    write_rows(output / "feature_reuse.csv", reuse_fields, reuse_rows)
    write_rows(output / "source_population_manifest.csv", list(source_manifest[0]), source_manifest)
    np.savez_compressed(
        output / "activation_selection_scatter_train.npz",
        activation=np.asarray([row["activation_frequency"] for row in predicate_rows if row["split"] == "train"]),
        selection=np.asarray([row["selection_frequency"] for row in predicate_rows if row["split"] == "train"]),
    )

    image_summary = finalize_image_values(image_values)
    summary = {
        "model": args.model,
        "environment": {
            "host": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "provenance": {
            "train_dir": str(args.train_dir.resolve()),
            "val_dir": str(args.val_dir.resolve()),
            "weights": str(args.weights.resolve()),
            "weights_sha256": sha256(args.weights),
            "phase3": str(args.phase3.resolve()),
            "phase3_sha256": sha256(args.phase3),
            "feature_dimension": dimension,
            "validation_argsort_kind": sort_kind or "numpy-default-quicksort",
        },
        "definitions": {
            "predicate": "(model, predicted class, signed physical feature, class-specific threshold)",
            "positive_activation": "activation >= minimum selected-source activation for that class/predicate",
            "negative_activation": "activation <= maximum selected-source activation for that class/predicate",
            "selection": "signed feature occurs in the extracted prediction-supporting pathway",
            "train_population": "source rows retained by majority-prediction selection and present in the initial-pathway checkpoint",
            "validation_population": "all held-out rows, grouped by model-predicted class; pathways conditioned on that prediction",
        },
        "population": {
            "raw_train_rows": raw_train_rows,
            "majority_rows_before_dictionary_overwrite": pre_overwrite_rows,
            "phase3_source_rows": global_counts["train_images"],
            "phase3_classes": len(pathways),
            "raw_validation_rows": raw_val_rows,
            "validation_label_correct": val_correct,
            "validation_label_accuracy": val_correct / raw_val_rows,
            "validation_analyzed_rows": global_counts["val_analyzed_images"],
            "validation_missing_threshold_classes": val_missing_threshold_classes,
        },
        "occurrences": dict(global_counts),
        "rates": {
            "train_selected_satisfies_threshold": global_counts["train_joint_occurrences"]
            / max(global_counts["train_selection_occurrences"], 1),
            "validation_selected_satisfies_threshold_among_eligible": global_counts["val_joint_occurrences"]
            / max(global_counts["val_selection_eligible_occurrences"], 1),
            "validation_selection_eligible_fraction": global_counts["val_selection_eligible_occurrences"]
            / max(
                global_counts["val_selection_eligible_occurrences"]
                + global_counts["val_selection_outside_vocabulary"],
                1,
            ),
        },
        "image_distributions": image_summary,
        "representative_concentration_classes": representatives,
        "class_metric_distributions": {
            split: {
                metric: describe([float(row[metric]) for row in class_rows if row["split"] == split])
                for metric in (
                    "vocabulary_size_signed", "vocabulary_size_physical", "activation_n80_fraction",
                    "selection_n80_fraction", "activation_observed", "selection_observed",
                )
            }
            for split in ("train", "val")
        },
        "runtime_seconds": time.time() - started,
    }
    json_dump(output / "summary.json", summary)
    print(json.dumps({"population": summary["population"], "rates": summary["rates"]}, indent=2), flush=True)
    return 0


def read_csv(path: Path) -> list[dict]:
    kwargs = {"mode": "rt", "newline": "", "encoding": "utf-8"}
    handle_context = gzip.open(path, **kwargs) if path.suffix == ".gz" else path.open(**kwargs)
    with handle_context as handle:
        return list(csv.DictReader(handle))


def analyze_all(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    models = args.models
    summaries = {model: json.loads((args.input_dir / model / "summary.json").read_text()) for model in models}

    summary_rows = []
    for model, data in summaries.items():
        image = data["image_distributions"]
        classes = data["class_metric_distributions"]["train"]
        summary_rows.append(
            {
                "model": model,
                "feature_dimension": data["provenance"]["feature_dimension"],
                "phase3_source_rows": data["population"]["phase3_source_rows"],
                "validation_rows": data["population"]["raw_validation_rows"],
                "validation_accuracy": data["population"]["validation_label_accuracy"],
                "median_class_signed_vocabulary": classes["vocabulary_size_signed"]["median"],
                "median_train_active_signed": image["train_all"]["active_signed"]["median"],
                "median_train_selected_signed": image["train_all"]["selected_signed"]["median"],
                "median_val_active_signed": image["val_all"]["active_signed"]["median"],
                "median_val_selected_signed": image["val_all"]["selected_signed"]["median"],
                "train_selected_satisfies_threshold": data["rates"]["train_selected_satisfies_threshold"],
                "val_selected_satisfies_threshold": data["rates"]["validation_selected_satisfies_threshold_among_eligible"],
                "val_selection_eligible_fraction": data["rates"]["validation_selection_eligible_fraction"],
                "median_selection_n80_fraction": classes["selection_n80_fraction"]["median"],
                "median_activation_n80_fraction": classes["activation_n80_fraction"]["median"],
            }
        )
    write_rows(output / "model_summary_table.csv", list(summary_rows[0]), summary_rows)

    distribution_rows = []
    activation_selection_cases = []
    distribution_metrics = [
        "activation_ge_1pct", "activation_ge_5pct", "activation_ge_10pct",
        "activation_ge_25pct", "selection_ge_1pct", "selection_ge_5pct",
        "selection_ge_10pct", "selection_ge_25pct", "activation_n50",
        "activation_n80", "activation_n90", "selection_n50", "selection_n80",
        "selection_n90", "activation_n80_fraction", "selection_n80_fraction",
    ]
    for model in models:
        class_data = read_csv(args.input_dir / model / "class_metrics.csv")
        for split in ("train", "val"):
            split_rows = [row for row in class_data if row["split"] == split]
            for metric in distribution_metrics:
                values = np.asarray([float(row[metric]) for row in split_rows])
                distribution_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "metric": metric,
                        "classes": len(values),
                        "q1": float(np.percentile(values, 25)),
                        "median": float(np.percentile(values, 50)),
                        "q3": float(np.percentile(values, 75)),
                    }
                )
        predicate_data = read_csv(args.input_dir / model / "predicate_metrics.csv.gz")
        candidates = [
            row for row in predicate_data
            if row["split"] == "train" and float(row["activation_frequency"]) >= 0.25
        ]
        candidates.sort(
            key=lambda row: float(row["activation_frequency"])
            - float(row["selection_frequency"]),
            reverse=True,
        )
        for rank, row in enumerate(candidates[:8], start=1):
            activation_selection_cases.append(
                {
                    **row,
                    "rank_within_model": rank,
                    "activation_selection_gap": float(row["activation_frequency"])
                    - float(row["selection_frequency"]),
                }
            )
    write_rows(output / "class_distribution_summary.csv", list(distribution_rows[0]), distribution_rows)
    write_rows(
        output / "activation_selection_cases.csv",
        list(activation_selection_cases[0]),
        activation_selection_cases,
    )

    # Figure 1: predicate activation versus selection (class-specific predicates).
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, model in zip(axes.flat, models):
        values = np.load(args.input_dir / model / "activation_selection_scatter_train.npz")
        x, y = values["activation"], values["selection"]
        if len(x) > 150000:
            rng = np.random.default_rng(20260913)
            keep = rng.choice(len(x), 150000, replace=False)
            x, y = x[keep], y[keep]
        hb = ax.hexbin(x, y, gridsize=55, bins="log", mincnt=1, cmap="viridis")
        ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
        ax.set(title=model, xlabel="Activation frequency", ylabel="Pathway-selection frequency", xlim=(0, 1), ylim=(0, 1))
        fig.colorbar(hb, ax=ax, label="log10 predicate count")
    fig.suptitle("Class-specific predicate activation versus pathway selection")
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"01_activation_vs_selection.{ext}", dpi=220)
    plt.close(fig)

    # Figure 2: low/median/high concentration classes selected by fixed score quantiles.
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True
    )
    colors = {"low": "#377eb8", "median": "#4daf4a", "high": "#e41a1c"}
    for ax, model in zip(axes.flat, models):
        reps = summaries[model]["representative_concentration_classes"]
        for level in ("low", "median", "high"):
            item = reps[level]
            for kind, linestyle in (("selection", "-"), ("activation", "--")):
                data = item[kind]
                ax.plot(data["predicate_fraction"], data["occurrence_share"], linestyle,
                        color=colors[level], label=f"{level} {kind}; c={item['class_id']}")
        ax.set(title=model, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.suptitle("Representative within-class concentration curves")
    fig.supxlabel("Fraction of observed predicates (ranked)")
    fig.supylabel("Cumulative occurrence share")
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"02_concentration_curves.{ext}", dpi=220)
    plt.close(fig)

    # Figure 3: median and IQR of per-image active/selected counts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, split in zip(axes, ("train_all", "val_all")):
        positions, labels = [], []
        for index, model in enumerate(models):
            for offset, metric, color in ((-0.13, "active_signed", "#377eb8"), (0.13, "selected_signed", "#e41a1c")):
                stat = summaries[model]["image_distributions"][split][metric]
                x = index + offset
                ax.vlines(x, stat["q1"], stat["q3"], color=color, linewidth=6)
                ax.plot(x, stat["median"], "o", color="black", markersize=4)
                ax.vlines(x, stat["p05"], stat["p95"], color=color, linewidth=1)
            positions.append(index); labels.append(model)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("Signed-predicate count per image")
        ax.set_title("Source/checkpoint" if split == "train_all" else "Held-out validation")
        ax.plot([], [], color="#377eb8", linewidth=6, label="active")
        ax.plot([], [], color="#e41a1c", linewidth=6, label="selected")
        ax.legend()
    fig.suptitle("Per-image counts: median, IQR, and 5th–95th percentiles")
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"03_per_image_counts.{ext}", dpi=220)
    plt.close(fig)

    # Figure 4: threshold validity and vocabulary eligibility on held-out pathways.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(models)); width = 0.34
    valid = [summaries[m]["rates"]["validation_selected_satisfies_threshold_among_eligible"] for m in models]
    eligible = [summaries[m]["rates"]["validation_selection_eligible_fraction"] for m in models]
    ax.bar(x - width / 2, valid, width, label="Selected and threshold-active / eligible selected")
    ax.bar(x + width / 2, eligible, width, label="Eligible selected / all selected")
    ax.set_xticks(x, models); ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction")
    ax.set_title("Held-out pathway threshold validity and training-vocabulary eligibility")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"04_heldout_threshold_validity.{ext}", dpi=220)
    plt.close(fig)

    # Figure 5: cross-class reuse of signed feature identities.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    candidate_rows = []
    for ax, model in zip(axes.flat, models):
        rows = read_csv(args.input_dir / model / "feature_reuse.csv")
        support = np.asarray([int(row["classes_selection_ge_1pct"]) for row in rows])
        ax.hist(support, bins=40, color="#984ea3", alpha=0.85)
        ax.set_yscale("log")
        ax.set(title=model, xlabel="Classes with selection frequency ≥1%", ylabel="Signed features (log count)")
        ranked = sorted(
            rows,
            key=lambda row: (
                int(row["classes_selection_ge_1pct"]) * float(row["std_class_selection_frequency"]),
                int(row["classes_selection_ge_1pct"]),
            ),
            reverse=True,
        )[:8]
        candidate_rows.extend(ranked)
    fig.suptitle("Cross-class reuse of signed physical-feature identities")
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"05_cross_class_reuse.{ext}", dpi=220)
    plt.close(fig)
    write_rows(output / "reuse_candidates.csv", list(candidate_rows[0]), candidate_rows)

    json_dump(
        output / "combined_summary.json",
        {
            "models": summaries,
            "summary_table": summary_rows,
            "class_distribution_summary": distribution_rows,
            "activation_selection_cases": activation_selection_cases,
        },
    )
    print(json.dumps(summary_rows, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--model", choices=sorted(MODEL_HEADS), required=True)
    model.add_argument("--train-dir", type=Path, required=True)
    model.add_argument("--val-dir", type=Path, required=True)
    model.add_argument("--weights", type=Path, required=True)
    model.add_argument("--phase3", type=Path, required=True)
    model.add_argument("--output-dir", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument("--input-dir", type=Path, required=True)
    combined.add_argument("--output-dir", type=Path, required=True)
    combined.add_argument("--models", nargs="+", default=["vit", "resnet", "convnext", "swin"])
    args = parser.parse_args()
    return analyze_model(args) if args.command == "model" else analyze_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
