"""Extend the frozen Score-CAM random-100 evaluation by one cutoff (0.10).

The original 0.60..0.15 outputs are read-only. Stored heatmaps are reused and
only previously failed arms are evaluated at 0.10. Results are written to a
separate directory.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import evaluate_grounding_random100 as base
from core import FeatureModel, compose_edit


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OLD = ROOT / "results/grounding_random100_four_models_v1"
OUT = ROOT / "results/scorecam_random100_four_models_lower010_v1"
WEIGHTS = Path(os.environ.get("VISIONLOGIC_WEIGHTS", WORKSPACE / "data/weights"))
MODELS = ("vit", "resnet", "convnext", "swin")
PIXELS = 224 * 224


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def active(q, value):
    return value >= q["threshold"] if q["branch"] == "+" else value <= q["threshold"]


def margin(q, value):
    return value - q["threshold"] if q["branch"] == "+" else q["threshold"] - value


def extension_proposals(heat, seed):
    original = base.CUTOFFS
    try:
        base.CUTOFFS = original + [0.10]
        guided, random = base.proposal_schedule(heat, seed)
    finally:
        base.CUTOFFS = original
    assert len(guided) == len(random) == 11
    return guided[-1], random[-1]


def evaluate_one(model, q, rgb, blur, proposal):
    mask = base.union_mask(proposal["boxes"])
    assert int(mask.sum()) == proposal["pixels"]
    attempt = {
        "cutoff": 0.10,
        "pixels": proposal["pixels"],
        "activation": None,
        "flipped": False,
        "evaluated": False,
    }
    if mask.any():
        edited = compose_edit(rgb, blur, mask)
        values, prediction = model.evaluate(Image.fromarray(edited))
        value = float(values[q["physical_id"]])
        attempt.update(
            activation=value,
            flipped=not active(q, value),
            evaluated=True,
            edited_prediction=prediction,
            signed_margin=margin(q, value),
            edited_rgb_sha256=hashlib.sha256(edited.tobytes()).hexdigest(),
        )
    return attempt


def extend_arm(old_arm, attempt):
    result = copy.deepcopy(old_arm)
    if result["success"]:
        return result
    result["attempts"].append(attempt)
    result["model_evaluations"] += int(attempt["evaluated"])
    if attempt["flipped"]:
        result["success"] = True
        result["accepted_step"] = 10
        result["accepted_area_pixels"] = attempt["pixels"]
    return result


def run(model_name):
    dest = OUT / model_name
    dest.mkdir(parents=True, exist_ok=True)
    model = FeatureModel(model_name, WEIGHTS, "cuda")
    records = sorted((OLD / model_name).glob("ILSVRC2012_val_*.json"))
    extensions = 0
    for i, old_path in enumerate(records, 1):
        out_path = dest / old_path.name
        if out_path.exists():
            continue
        old = load(old_path)
        folder = OLD / model_name / old["image_id"]
        image = Image.open(folder / "input.png").convert("RGB")
        rgb = np.asarray(image)
        blur = np.asarray(image.filter(ImageFilter.GaussianBlur(20)))
        result = {k: copy.deepcopy(v) for k, v in old.items() if k != "predicates"}
        result["predicates"] = []
        if not old["predicates"]:
            write(out_path, result)
            print(f"{model_name} {i}/{len(records)} {old['image_id']} targets=0", flush=True)
            continue
        maps = np.load(folder / "heatmaps.npz")
        for p in old["predicates"]:
            key = f"p{p['predicate']['logical_id']}"
            heat = maps[key]
            guided_prop, random_prop = extension_proposals(heat, int(p["control_seed"]))
            guided_attempt = evaluate_one(model, p["predicate"], rgb, blur, guided_prop) if not p["guided"]["success"] else None
            random_attempt = evaluate_one(model, p["predicate"], rgb, blur, random_prop) if not p["random"]["success"] else None
            lower010 = {
                "guided": extend_arm(p["guided"], guided_attempt) if guided_attempt is not None else copy.deepcopy(p["guided"]),
                "random": extend_arm(p["random"], random_attempt) if random_attempt is not None else copy.deepcopy(p["random"]),
            }
            if lower010["guided"]["success"] and not p["guided"]["success"]:
                extensions += 1
            result["predicates"].append({
                "predicate": copy.deepcopy(p["predicate"]),
                "original_activation": p["original_activation"],
                "zero_heatmap": p["zero_heatmap"],
                "control_seed": p["control_seed"],
                "lower015": {"guided": copy.deepcopy(p["guided"]), "random": copy.deepcopy(p["random"])},
                "lower010": lower010,
                "cutoff010_geometry": {"guided": guided_prop, "random": random_prop},
            })
        write(out_path, result)
        print(f"{model_name} {i}/{len(records)} {old['image_id']} targets={len(old['predicates'])}", flush=True)

    all_records = [load(p) for p in sorted(dest.glob("ILSVRC2012_val_*.json"))]
    targets = [p for r in all_records for p in r["predicates"]]
    summary = {
        "images": len(all_records),
        "eligible_images": sum(bool(r["predicates"]) for r in all_records),
        "targets": len(targets),
        "new_guided_successes_at_010": sum(p["lower010"]["guided"]["success"] and not p["lower015"]["guided"]["success"] for p in targets),
    }
    for policy in ("lower015", "lower010"):
        summary[policy] = {}
        for arm in ("guided", "random"):
            passed = [p for p in targets if p[policy][arm]["success"]]
            areas = [p[policy][arm]["accepted_area_pixels"] / PIXELS * 100 for p in passed]
            images = {r["image_id"] for r in all_records if any(p[policy][arm]["success"] for p in r["predicates"])}
            summary[policy][arm] = {
                "passed": len(passed), "total": len(targets), "rate": len(passed) / len(targets),
                "median_area_percent": float(np.median(areas)),
                "area_iqr_percent": [float(x) for x in np.percentile(areas, [25, 75])],
                "images_with_success": len(images),
            }
    write(dest / "summary.json", summary)
    write(dest / "configuration.json", {
        "model": model_name,
        "source": str(OLD / model_name),
        "method": "Frozen neuron-targeted Score-CAM heatmaps extended by one 0.10 cutoff",
        "cutoffs": base.CUTOFFS + [0.10],
        "blur_radius": 20,
        "original_results_modified": False,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    run(parser.parse_args().model)
