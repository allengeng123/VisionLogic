"""Verify classifier accuracy reconstructed from saved penultimate activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


HEAD_KEYS = {
    "vit": ("heads.head.weight", "heads.head.bias"),
    "resnet": ("fc.weight", "fc.bias"),
    "convnext": ("classifier.2.weight", "classifier.2.bias"),
    "swin": ("head.weight", "head.bias"),
}

CONSENSUS_RE = re.compile(
    r"Key '([^']+)' -> Discovered Class (\d+) \(Consensus: ([0-9.]+)%\)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notebook_consensus(path: Path) -> dict[str, tuple[int, float]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    documented: dict[str, tuple[int, float]] = {}
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            for filename, class_id, percent in CONSENSUS_RE.findall(str(text)):
                documented[filename] = (int(class_id), float(percent))
    return documented


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(HEAD_KEYS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, help="Optional historical notebook transcript for a consensus audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    weight_key, bias_key = HEAD_KEYS[args.model]
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    weights = state[weight_key].detach().cpu().numpy()
    bias = state[bias_key].detach().cpu().numpy()
    documented = notebook_consensus(args.notebook) if args.notebook else {}
    files = sorted(
        path
        for path in args.data_dir.iterdir()
        if path.suffix.lower() in {".pkl", ".pickle"}
    )
    if len(files) != 1000:
        raise RuntimeError(f"Expected 1000 activation files, found {len(files)}")

    total = 0
    top1_correct = 0
    top5_correct = 0
    majority_total = 0
    class_top1_rates = []
    majority_matches_label = 0
    majority_label_mismatches = []
    documented_mismatches = []
    undocumented_files = []
    shapes = set()

    for label, path in enumerate(files):
        with path.open("rb") as handle:
            matrix = pickle.load(handle)
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise TypeError(f"Unexpected activation payload in {path}")
        shapes.add(tuple(int(value) for value in matrix.shape))
        logits = matrix @ weights.T + bias
        predictions = np.argmax(logits, axis=1)
        label_count = int(np.count_nonzero(predictions == label))
        top1_correct += label_count
        total += len(predictions)
        class_top1_rates.append(label_count / len(predictions))

        top5 = np.argpartition(logits, kth=-5, axis=1)[:, -5:]
        top5_correct += int(np.count_nonzero(np.any(top5 == label, axis=1)))

        values, counts = np.unique(predictions, return_counts=True)
        majority_index = int(np.argmax(counts))
        majority_class = int(values[majority_index])
        majority_count = int(counts[majority_index])
        majority_total += majority_count
        majority_matches_label += int(majority_class == label)
        if majority_class != label:
            majority_label_mismatches.append(
                {
                    "file": path.name,
                    "filename_label": label,
                    "majority_predicted_class": majority_class,
                    "majority_count": majority_count,
                    "filename_label_count": label_count,
                    "row_count": len(predictions),
                }
            )

        expected = documented.get(path.name)
        actual_display_percent = float(f"{100 * majority_count / len(predictions):.1f}")
        if expected is None:
            undocumented_files.append(path.name)
        elif expected != (majority_class, actual_display_percent):
            if len(documented_mismatches) < 50:
                documented_mismatches.append(
                    {
                        "file": path.name,
                        "documented_class": expected[0],
                        "actual_class": majority_class,
                        "documented_consensus_percent": expected[1],
                        "actual_consensus_percent": actual_display_percent,
                    }
                )
        if (label + 1) % 50 == 0:
            print(f"{args.model}: processed {label + 1}/1000", flush=True)

    documented_extra = sorted(set(documented) - {path.name for path in files})
    summary = {
        "model": args.model,
        "environment": {
            "host": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "inputs": {
            "data_dir": str(args.data_dir.resolve()),
            "file_count": len(files),
            "activation_shapes": sorted(shapes),
            "weights": str(args.weights.resolve()),
            "weights_sha256": sha256(args.weights),
            "notebook": str(args.notebook.resolve()) if args.notebook else None,
            "notebook_sha256": sha256(args.notebook) if args.notebook else None,
        },
        "label_policy": "zero-based class index from lexicographically sorted ImageNet synset filenames",
        "prediction_policy": "argmax(X @ W.T + b); top-5 from the five largest logits",
        "accuracy": {
            "denominator": total,
            "top1_correct": top1_correct,
            "top1_accuracy": top1_correct / total,
            "top5_correct": top5_correct,
            "top5_accuracy": top5_correct / total,
            "macro_class_top1_accuracy": float(np.mean(class_top1_rates)),
        },
        "notebook_consensus": {
            "transcript_comparison_performed": args.notebook is not None,
            "definition": "per-file fraction assigned to that file's majority predicted class",
            "majority_selected_rows": majority_total,
            "micro_consensus_rate": majority_total / total,
            "majority_class_matches_filename_label": majority_matches_label,
            "majority_label_mismatches": majority_label_mismatches,
            "documented_file_count": len(documented),
            "matched_documented_files": len(documented)
            - len(documented_mismatches)
            - len(documented_extra),
            "mismatch_count": len(documented_mismatches),
            "first_mismatches": documented_mismatches,
            "activation_files_without_documented_output": undocumented_files,
            "documented_outputs_without_activation_file": documented_extra,
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["accuracy"], indent=2), flush=True)
    print(json.dumps(summary["notebook_consensus"], indent=2), flush=True)
    # Missing saved output lines are an incomplete notebook transcript, not a
    # numerical mismatch; compare every documented line that is available.
    return 0 if not documented_mismatches and not documented_extra else 2


if __name__ == "__main__":
    raise SystemExit(main())
