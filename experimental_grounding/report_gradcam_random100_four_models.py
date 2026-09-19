import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experimental_grounding/results/grounding_random100_four_models_v1"
NEW = ROOT / "experimental_grounding/results/gradcam_random100_four_models_lower010_v1"
SCORE010 = ROOT / "experimental_grounding/results/scorecam_random100_four_models_lower010_v1"
OUT = ROOT / "output"
MODELS = ["vit", "resnet", "convnext", "swin"]
NAMES = {"vit": "ViT-B/16", "resnet": "ResNet-50", "convnext": "ConvNeXt-B", "swin": "Swin-B"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(folder):
    return {
        p.stem: load(p)
        for p in folder.glob("ILSVRC2012_val_*.json")
    }


def predicate_key(image_id, p):
    q = p["predicate"]
    return image_id, q["logical_id"], q["branch"]


old_summary = load(OLD / "summary.json")["models"]
new_summary = {m: load(NEW / m / "summary.json") for m in MODELS}
score010_summary = {m: load(SCORE010 / m / "summary.json") for m in MODELS}
paired = {}
sequential = {}
occupancy = {}
audit = {
    "passed": True,
    "models": {},
    "total_targets": 0,
    "metadata_mismatches": 0,
    "lower015_prefix_mismatches": 0,
    "scorecam_lower015_prefix_mismatches": 0,
    "invalid_areas": 0,
}

for model in MODELS:
    old_recs = records(OLD / model)
    new_recs = records(NEW / model)
    score010_recs = records(SCORE010 / model)
    old_targets, new_targets, score010_targets = {}, {}, {}
    for image_id, rec in old_recs.items():
        for p in rec["predicates"]:
            old_targets[predicate_key(image_id, p)] = p
    for image_id, rec in new_recs.items():
        for p in rec["predicates"]:
            new_targets[predicate_key(image_id, p)] = p
    for image_id, rec in score010_recs.items():
        for p in rec["predicates"]:
            score010_targets[predicate_key(image_id, p)] = p

    keys_match = set(old_targets) == set(new_targets) == set(score010_targets)
    metadata_mismatches = 0
    prefix_mismatches = 0
    score_prefix_mismatches = 0
    invalid_areas = 0
    outcomes = {
        "score_vs_grad015": {"both": 0, "score_only": 0, "grad_only": 0, "neither": 0},
        "score_vs_grad010": {"both": 0, "score_only": 0, "grad_only": 0, "neither": 0},
        "score010_vs_grad010": {"both": 0, "score_only": 0, "grad_only": 0, "neither": 0},
    }
    seq_guided = 0
    seq_random = 0
    fallback_tests = 0
    fallback_successes = 0
    seq_areas = []
    seq_random_areas = []
    score015_areas = []
    score010_areas = []
    grad010_areas = []
    seq_guided_images = set()
    seq_random_images = set()
    for key in sorted(set(old_targets) & set(new_targets)):
        old_p, new_p, score010_p = old_targets[key], new_targets[key], score010_targets[key]
        oq, nq = old_p["predicate"], new_p["predicate"]
        if (
            oq != nq
            or oq != score010_p["predicate"]
            or abs(old_p["original_activation"] - new_p["original_activation"]) > 1e-6
            or abs(old_p["original_activation"] - score010_p["original_activation"]) > 1e-6
        ):
            metadata_mismatches += 1

        for arm in ("guided", "random"):
            if score010_p["lower015"][arm] != old_p[arm]:
                score_prefix_mismatches += 1
            a = new_p["lower015"][arm]
            b = new_p["lower010"][arm]
            if a["success"] != (b["success"] and b["accepted_step"] is not None and b["accepted_step"] <= 9):
                prefix_mismatches += 1
            for trial in (a, b):
                area = trial["accepted_area_pixels"]
                if area is not None and not (0 < area <= 224 * 224):
                    invalid_areas += 1

        score = old_p["guided"]["success"]
        if score:
            score015_areas.append(old_p["guided"]["accepted_area_pixels"] / (224 * 224) * 100)
        if score010_p["lower010"]["guided"]["success"]:
            score010_areas.append(score010_p["lower010"]["guided"]["accepted_area_pixels"] / (224 * 224) * 100)
        if new_p["lower010"]["guided"]["success"]:
            grad010_areas.append(new_p["lower010"]["guided"]["accepted_area_pixels"] / (224 * 224) * 100)
        for label, policy in (("score_vs_grad015", "lower015"), ("score_vs_grad010", "lower010")):
            grad = new_p[policy]["guided"]["success"]
            bucket = "both" if score and grad else "score_only" if score else "grad_only" if grad else "neither"
            outcomes[label][bucket] += 1
        score010_success = score010_p["lower010"]["guided"]["success"]
        grad010_success = new_p["lower010"]["guided"]["success"]
        bucket = "both" if score010_success and grad010_success else "score_only" if score010_success else "grad_only" if grad010_success else "neither"
        outcomes["score010_vs_grad010"][bucket] += 1

        # Deterministic recovery policy: Grad-CAM to 0.10 first, then the
        # frozen Score-CAM-to-0.10 proposal only if Grad-CAM fails.  The
        # random control receives the same two-proposal opportunity.
        grad_g = new_p["lower010"]["guided"]["success"]
        score_g = score010_p["lower010"]["guided"]["success"]
        grad_r = new_p["lower010"]["random"]["success"]
        score_r = score010_p["lower010"]["random"]["success"]
        if grad_g:
            seq_guided += 1
            seq_guided_images.add(key[0])
            seq_areas.append(new_p["lower010"]["guided"]["accepted_area_pixels"] / (224 * 224) * 100)
        else:
            fallback_tests += 1
            if score_g:
                fallback_successes += 1
                seq_guided += 1
                seq_guided_images.add(key[0])
                seq_areas.append(score010_p["lower010"]["guided"]["accepted_area_pixels"] / (224 * 224) * 100)
        if grad_r or score_r:
            seq_random += 1
            seq_random_images.add(key[0])
            if grad_r:
                seq_random_areas.append(new_p["lower010"]["random"]["accepted_area_pixels"] / (224 * 224) * 100)
            else:
                seq_random_areas.append(score010_p["lower010"]["random"]["accepted_area_pixels"] / (224 * 224) * 100)

    model_audit = {
        "old_images": len(old_recs),
        "new_images": len(new_recs),
        "score010_images": len(score010_recs),
        "old_targets": len(old_targets),
        "new_targets": len(new_targets),
        "score010_targets": len(score010_targets),
        "target_keys_match": keys_match,
        "metadata_mismatches": metadata_mismatches,
        "lower015_prefix_mismatches": prefix_mismatches,
        "scorecam_lower015_prefix_mismatches": score_prefix_mismatches,
        "invalid_areas": invalid_areas,
    }
    audit["models"][model] = model_audit
    audit["total_targets"] += len(new_targets)
    audit["metadata_mismatches"] += metadata_mismatches
    audit["lower015_prefix_mismatches"] += prefix_mismatches
    audit["scorecam_lower015_prefix_mismatches"] += score_prefix_mismatches
    audit["invalid_areas"] += invalid_areas
    if not keys_match or metadata_mismatches or prefix_mismatches or score_prefix_mismatches or invalid_areas:
        audit["passed"] = False
    paired[model] = outcomes
    sequential[model] = {
        "targets": len(new_targets),
        "guided_successes": seq_guided,
        "guided_rate": seq_guided / len(new_targets),
        "mirrored_random_successes": seq_random,
        "mirrored_random_rate": seq_random / len(new_targets),
        "guided_minus_random": (seq_guided - seq_random) / len(new_targets),
        "gradcam_failures_sent_to_fallback": fallback_tests,
        "scorecam_fallback_successes": fallback_successes,
        "median_accepted_area_percent": float(np.median(seq_areas)),
        "accepted_area_iqr_percent": [float(x) for x in np.percentile(seq_areas, [25, 75])],
        "median_random_area_percent": float(np.median(seq_random_areas)),
        "random_area_iqr_percent": [float(x) for x in np.percentile(seq_random_areas, [25, 75])],
        "images_with_guided_success": len(seq_guided_images),
        "images_with_random_success": len(seq_random_images),
    }
    occupancy[model] = {
        "scorecam015": {
            "median": float(np.median(score015_areas)),
            "iqr": [float(x) for x in np.percentile(score015_areas, [25, 75])],
        },
        "scorecam010": {
            "median": float(np.median(score010_areas)),
            "iqr": [float(x) for x in np.percentile(score010_areas, [25, 75])],
        },
        "gradcam010": {
            "median": float(np.median(grad010_areas)),
            "iqr": [float(x) for x in np.percentile(grad010_areas, [25, 75])],
        },
        "grad_first_score_fallback": {
            "median": float(np.median(seq_areas)),
            "iqr": [float(x) for x in np.percentile(seq_areas, [25, 75])],
        },
        "mirrored_random": {
            "median": float(np.median(seq_random_areas)),
            "iqr": [float(x) for x in np.percentile(seq_random_areas, [25, 75])],
        },
    }


def pct(value):
    return 100.0 * value


rows = []
for model in MODELS:
    old = old_summary[model]
    new = new_summary[model]
    conditions = [
        (
            "Score-CAM, 0.15",
            old["guided"]["success_percent"],
            old["random"]["success_percent"],
            old["guided"]["median_successful_area_percent"],
            old["zero_heatmap_targets"],
        ),
        (
            "Score-CAM, 0.10",
            pct(score010_summary[model]["lower010"]["guided"]["rate"]),
            pct(score010_summary[model]["lower010"]["random"]["rate"]),
            score010_summary[model]["lower010"]["guided"]["median_area_percent"],
            old["zero_heatmap_targets"],
        ),
        (
            "Grad-CAM, 0.15",
            pct(new["lower015"]["guided"]["rate"]),
            pct(new["lower015"]["random"]["rate"]),
            new["lower015"]["guided"]["median_area_percent"],
            new["zero_maps"],
        ),
        (
            "Grad-CAM, 0.10",
            pct(new["lower010"]["guided"]["rate"]),
            pct(new["lower010"]["random"]["rate"]),
            new["lower010"]["guided"]["median_area_percent"],
            new["zero_maps"],
        ),
    ]
    for condition, guided, random, area, zero in conditions:
        rows.append((model, condition, guided, random, guided - random, area, zero))


# Four-panel result figure.
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.25), sharey=True)
conditions = ["Score-CAM\n0.15", "Score-CAM\n0.10", "Grad-CAM\n0.15", "Grad-CAM\n0.10"]
for ax, model in zip(axes, MODELS):
    model_rows = [r for r in rows if r[0] == model]
    x = np.arange(4)
    guided = [r[2] for r in model_rows]
    random = [r[3] for r in model_rows]
    ax.bar(x - 0.18, guided, width=0.36, color="#2878B5", label="Guided")
    ax.bar(x + 0.18, random, width=0.36, color="#B8B8B8", label="Matched random")
    ax.set_title(NAMES[model], fontsize=10.5, weight="bold")
    ax.set_xticks(x, conditions, fontsize=7.7)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if model == "vit":
        ax.set_ylabel("Predicate deactivation success (%)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7.2, frameon=False)
fig.suptitle("Neuron attribution comparison on the same random-100 ImageNet sample", fontsize=11.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8)
OUT.mkdir(exist_ok=True)
figure_path = OUT / "gradcam_vs_scorecam_random100_four_models.png"
fig.savefig(figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)


table_lines = [
    "| Model | Condition | Guided | Matched random | Guided - random | Median guided area | Zero maps |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for model, condition, guided, random, gap, area, zero in rows:
    table_lines.append(
        f"| {NAMES[model]} | {condition} | {guided:.1f}% | {random:.1f}% | {gap:+.1f} pp | {area:.1f}% | {zero} |"
    )

paired_lines = [
    "| Model | Comparison | Both pass | Score-CAM only | Grad-CAM only | Neither |",
    "|---|---|---:|---:|---:|---:|",
]
for model in MODELS:
    for label, display in (("score_vs_grad015", "Score 0.15 vs Grad 0.15"), ("score010_vs_grad010", "Score 0.10 vs Grad 0.10")):
        x = paired[model][label]
        paired_lines.append(
            f"| {NAMES[model]} | {display} | {x['both']} | {x['score_only']} | {x['grad_only']} | {x['neither']} |"
        )

image_coverage_lines = [
    "| Model | Eligible images | Score-CAM, 0.15 | Score-CAM, 0.10 | Grad-CAM, 0.15 | Grad-CAM, 0.10 |",
    "|---|---:|---:|---:|---:|---:|",
]
for model in MODELS:
    old = old_summary[model]
    new = new_summary[model]
    denom = new["eligible_images"]
    score_n = old["guided"]["images_with_success"]
    score010_n = score010_summary[model]["lower010"]["guided"]["images_with_success"]
    grad015_n = new["lower015"]["guided"]["images_with_success"]
    grad010_n = new["lower010"]["guided"]["images_with_success"]
    image_coverage_lines.append(
        f"| {NAMES[model]} | {denom} | {score_n}/{denom} ({100*score_n/denom:.1f}%) | "
        f"{score010_n}/{denom} ({100*score010_n/denom:.1f}%) | "
        f"{grad015_n}/{denom} ({100*grad015_n/denom:.1f}%) | "
        f"{grad010_n}/{denom} ({100*grad010_n/denom:.1f}%) |"
    )

sequential_lines = [
    "| Model | Guided success | Mirrored two-proposal random | Gap | Images covered | Score-CAM recoveries | Guided occupancy | Random occupancy |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for model in MODELS:
    x = sequential[model]
    denom_images = new_summary[model]["eligible_images"]
    sequential_lines.append(
        f"| {NAMES[model]} | {x['guided_successes']}/{x['targets']} ({100*x['guided_rate']:.1f}%) | "
        f"{x['mirrored_random_successes']}/{x['targets']} ({100*x['mirrored_random_rate']:.1f}%) | "
        f"{100*x['guided_minus_random']:+.1f} pp | "
        f"{x['images_with_guided_success']}/{denom_images} ({100*x['images_with_guided_success']/denom_images:.1f}%) | "
        f"{x['scorecam_fallback_successes']}/{x['gradcam_failures_sent_to_fallback']} | "
        f"{x['median_accepted_area_percent']:.1f}% | {x['median_random_area_percent']:.1f}% |"
    )

occupancy_lines = [
    "| Model | Score-CAM 0.15 | Score-CAM 0.10 | Grad-CAM 0.10 | Combined guided | Combined random |",
    "|---|---:|---:|---:|---:|---:|",
]
for model in MODELS:
    x = occupancy[model]
    def med_iqr(item):
        return f"{item['median']:.1f}% [{item['iqr'][0]:.1f}, {item['iqr'][1]:.1f}]"
    occupancy_lines.append(
        f"| {NAMES[model]} | {med_iqr(x['scorecam015'])} | {med_iqr(x['scorecam010'])} | {med_iqr(x['gradcam010'])} | "
        f"{med_iqr(x['grad_first_score_fallback'])} | {med_iqr(x['mirrored_random'])} |"
    )

new010_total = sum(new_summary[m]["lower_bound_change"]["new_at_010"] for m in MODELS)
score010_total = sum(score010_summary[m]["new_guided_successes_at_010"] for m in MODELS)
report = f"""# Neuron-targeted Grad-CAM on the four-model random-100 grounding set

## Bottom line

Grad-CAM with a 0.10 lower cutoff is **not a uniform improvement over the existing Score-CAM procedure**. It is the correct practical replacement for the broken ConvNeXt Score-CAM configuration, gives a cleaner guided-versus-random separation and smaller regions on ResNet, but reduces raw guided success on ViT and fails the matched-random control on Swin. The evidence therefore supports an architecture-aware choice, not a blanket four-model switch.

## Controlled protocol

- Same fixed seed-42 random sample of 100 ImageNet validation images for every model.
- Same original model prediction, class-conditioned predicate, sign, threshold, and eligible pathway predicate set, totaling {audit['total_targets']} predicate tests.
- Neuron-targeted attribution uses the signed frozen-head input neuron, not the class logit.
- Same box-union proposal, Gaussian blur radius 20, one-way predicate-deactivation test, and area-matched random-region control.
- Both Score-CAM and Grad-CAM search the same cutoffs from 0.60 through 0.10. Their 0.15-prefix results are also reported to isolate the attribution change from the extra search step.

## Results

{chr(10).join(table_lines)}

The 0.10 step rescued {new010_total} additional Grad-CAM and {score010_total} additional Score-CAM predicate tests across the four models, but it also enlarged regions and generally raised the random-control pass rate. It should therefore not be interpreted as a free improvement.

### Image-level explanation coverage

An image is covered when at least one eligible pathway predicate has a guided region whose removal deactivates that predicate. The denominator is the number of sampled images with at least one eligible predicate, rather than always 100; ViT has 99 eligible images.

{chr(10).join(image_coverage_lines)}

This is a useful user-facing coverage measure, but it must be reported beside predicate-level success and matched-random controls. In particular, Swin reaches near-complete image coverage even though its Grad-CAM guided regions underperform matched random regions. Image coverage alone therefore does not establish localization specificity.

### Grad-CAM-first proposal recovery

The deterministic recovery policy first searches Grad-CAM proposals through 0.10 and invokes the frozen Score-CAM search through 0.10 only when Grad-CAM does not deactivate the predicate. To account for the second opportunity, the control below also succeeds if either method's area-matched random proposal succeeds. This is stricter than comparing the combined guided policy against only one random proposal.

{chr(10).join(sequential_lines)}

This policy raises explanation availability to 90--100% of eligible images, with complete coverage for ViT and Swin. Success and occupancy must be interpreted jointly. For ViT, ResNet, and ConvNeXt, guided proposals succeed more often despite having smaller median occupancy than successful random controls. For Swin, the random control succeeds more often while occupying less area, so the combined policy does not establish localization specificity there. The method order should be fixed before evaluation and the fallback rate should be disclosed.

### Accepted box occupancy

Occupancy is the fraction of the 224-by-224 input covered by the union of accepted connected-component boxes. Values are median percentages with the interquartile range in brackets, computed only among successful predicate tests. For the combined policy, Grad-CAM's accepted box is used when it succeeds; otherwise the accepted Score-CAM fallback box is used. The random column follows the same Grad-CAM-first order.

{chr(10).join(occupancy_lines)}

The combined policy does not obtain its higher coverage by expanding beyond both component methods. Its median occupancy remains close to Grad-CAM because Grad-CAM supplies most accepted proposals. Score-CAM fallback increases median occupancy for architectures where it recovers additional failures.

### Paired guided outcomes on identical predicates

{chr(10).join(paired_lines)}

## Interpretation by architecture

- **ViT-B/16:** Score-CAM remains stronger in raw guided success (92.0% versus 87.7%). Grad-CAM produces smaller successful regions (23.0% versus 33.1%) and a slightly larger guided-minus-random margin (6.9 versus 5.4 points). This is a compactness tradeoff, not a clear universal winner.
- **ResNet-50:** Score-CAM reaches higher raw success at 0.10 (84.5% versus 73.6%), while Grad-CAM reduces median successful area from 57.9% to 41.2% and increases the guided-minus-random gap from 7.0 to 18.6 points. This is the best evidence that Grad-CAM yields cleaner localization despite lower coverage.
- **ConvNeXt-B:** Grad-CAM eliminates the earlier 168 zero-map failures and reaches 74.8% guided success. This fixes the ConvNeXt attribution implementation. The 7.0-point advantage over matched random remains modest, so the raw gain over the broken Score-CAM result should not be presented as a general method comparison.
- **Swin-B:** Score-CAM at 0.10 reaches 83.9% guided success and narrowly exceeds its matched random rate of 83.7%; Grad-CAM remains worse at 65.2% guided versus 74.9% random. Grad-CAM should not replace Score-CAM for Swin under this layer and protocol.

## Recommendation

Do not change the paper to claim that Grad-CAM is the universal grounding method. The defensible configuration from these results is:

1. retain Score-CAM for ViT and Swin;
2. consider Grad-CAM for ResNet because it yields smaller regions and a stronger control gap;
3. use Grad-CAM for ConvNeXt because the prior Score-CAM implementation is invalid there;
4. if a single method is required for conceptual simplicity, more layer selection work is needed before Grad-CAM can be used on Swin.

Using 0.10 for both methods makes the recovery rule symmetric. The added step helps guided coverage but also increases matched-random success, so all claims should retain the mirrored two-proposal control.

## Audit

Audit passed: **{audit['passed']}**. The run contains exactly the same image/predicate keys and unchanged classes, signs, thresholds, and original activations as the frozen Score-CAM evaluation. The independently summarized 0.15 policies are exact prefixes of both 0.10 runs. Metadata mismatches: {audit['metadata_mismatches']}; Grad-CAM prefix mismatches: {audit['lower015_prefix_mismatches']}; Score-CAM prefix mismatches: {audit['scorecam_lower015_prefix_mismatches']}; invalid accepted areas: {audit['invalid_areas']}. Experimental outputs are separate; manuscript files and prior results were not modified.
"""

(OUT / "GradCAM_Random100_Four_Models_Report.md").write_text(report, encoding="utf-8")
(OUT / "DualCAM_Iterative_Random100_Four_Models_Report.md").write_text(report, encoding="utf-8")
(NEW / "cross_method_paired_outcomes.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
(NEW / "gradcam_first_scorecam_fallback.json").write_text(json.dumps(sequential, indent=2), encoding="utf-8")
(NEW / "box_occupancy_comparison.json").write_text(json.dumps(occupancy, indent=2), encoding="utf-8")
(NEW / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"audit": audit, "figure": str(figure_path), "report": str(OUT / 'GradCAM_Random100_Four_Models_Report.md')}, indent=2))
