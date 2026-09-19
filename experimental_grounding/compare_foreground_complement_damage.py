"""Decompose fixed box damage into BiRefNet foreground and box complement.

For every predicate in the fresh 30-image blur-only ResNet demo, this script
reuses the exact terminal tested box and its saved BiRefNet display mask. The
box is partitioned into two disjoint regions: foreground intersection and box
complement. Radius-20 blur is applied to each region separately. No heatmap,
threshold, box, segmentation, or predicate selection is rerun.
"""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

from core import FeatureModel, compose_edit, digest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/box_only_resnet30_bird_squirrel_church_fox_blur20_initialpaths_v2"
OUT = ROOT / "results/foreground_vs_box_complement_blur20_resnet30_v1"
WEIGHTS = Path(os.environ.get("VISIONLOGIC_WEIGHTS", ROOT.parent / "data/weights"))
BLUR_RADIUS = 20
TOLERANCE = 1e-7


def is_active(branch: str, threshold: float, value: float) -> bool:
    return value >= threshold if branch == "+" else value <= threshold


def signed_margin(branch: str, threshold: float, value: float) -> float:
    return value - threshold if branch == "+" else threshold - value


def damage(original_margin: float, edited_margin: float) -> float:
    return original_margin - edited_margin


def outline(rgb: np.ndarray, foreground: np.ndarray, complement: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    kernel = np.ones((5, 5), np.uint8)
    for mask, color in ((foreground, (22, 163, 74)), (complement, (220, 38, 38))):
        outer = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        inner = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        result[outer & ~inner] = np.array(color, dtype=np.uint8)
    return result


def subset_summary(cases: list[dict]) -> dict:
    fg_only = sum(c["foreground"]["flipped"] and not c["complement"]["flipped"] for c in cases)
    comp_only = sum(c["complement"]["flipped"] and not c["foreground"]["flipped"] for c in cases)
    both = sum(c["foreground"]["flipped"] and c["complement"]["flipped"] for c in cases)
    neither = len(cases) - fg_only - comp_only - both
    return {
        "tested": len(cases),
        "full_box_flipped": sum(c["full_box"]["flipped"] for c in cases),
        "foreground_flipped": sum(c["foreground"]["flipped"] for c in cases),
        "complement_flipped": sum(c["complement"]["flipped"] for c in cases),
        "foreground_only_flipped": fg_only,
        "complement_only_flipped": comp_only,
        "both_partition_members_flipped": both,
        "neither_partition_member_flipped": neither,
        "foreground_larger_damage": sum(c["comparison"] == "foreground" for c in cases),
        "complement_larger_damage": sum(c["comparison"] == "complement" for c in cases),
        "tied_damage": sum(c["comparison"] == "tie" for c in cases),
        "foreground_positive_damage": sum(c["foreground"]["damage"] > TOLERANCE for c in cases),
        "complement_positive_damage": sum(c["complement"]["damage"] > TOLERANCE for c in cases),
        "foreground_negative_damage": sum(c["foreground"]["damage"] < -TOLERANCE for c in cases),
        "complement_negative_damage": sum(c["complement"]["damage"] < -TOLERANCE for c in cases),
    }


def render(data: dict) -> None:
    all_s = data["summary"]["all_terminal_boxes"]
    acc_s = data["summary"]["validated_full_boxes"]
    summary_rows = []
    for label, item in (("All terminal boxes", all_s), ("Validated full boxes", acc_s)):
        summary_rows.append(
            f"<tr><td>{label}</td><td>{item['tested']}</td><td>{item['foreground_larger_damage']}</td>"
            f"<td>{item['complement_larger_damage']}</td><td>{item['tied_damage']}</td>"
            f"<td>{item['foreground_flipped']}</td><td>{item['complement_flipped']}</td></tr>"
        )

    cards = []
    for case in data["cases"]:
        predicate = case["predicate"]
        symbol = "≥" if predicate["branch"] == "+" else "≤"
        winner = case["comparison"]
        cards.append(
            f'''<article data-valid="{str(case['source_accepted']).lower()}" data-winner="{winner}">
            <h2>{html.escape(case['image_name'])} · p<sub>({predicate['physical_id']},{predicate['branch']})</sub> · {html.escape(winner)} larger damage</h2>
            <p>Full box {'validated' if case['source_accepted'] else 'did not flip'} at cutoff {case['cutoff']:.2f}. Original z={case['original_activation']:.5g}; active when z {symbol} {predicate['threshold']:.5g}. Foreground covers {case['foreground_pixels']/case['box_pixels']:.1%} of the box.</p>
            <div class="images">
              <figure><img loading="lazy" src="{case['key']}/regions.png"><figcaption>Green: foreground intersection<br>Red: box complement</figcaption></figure>
              <figure><img loading="lazy" src="{case['key']}/full_box.png"><figcaption>Full box · D={case['full_box']['damage']:.4g} · {'FLIP' if case['full_box']['flipped'] else 'active'}</figcaption></figure>
              <figure><img loading="lazy" src="{case['key']}/foreground.png"><figcaption>Foreground · D={case['foreground']['damage']:.4g} · {'FLIP' if case['foreground']['flipped'] else 'active'}</figcaption></figure>
              <figure><img loading="lazy" src="{case['key']}/complement.png"><figcaption>Complement · D={case['complement']['damage']:.4g} · {'FLIP' if case['complement']['flipped'] else 'active'}</figcaption></figure>
            </div>
            <table><thead><tr><th>Region blurred</th><th>Pixels</th><th>Post-ablation z</th><th>Signed margin damage</th><th>Flip</th></tr></thead><tbody>
              <tr><td>Full box</td><td>{case['box_pixels']}</td><td>{case['full_box']['activation']:.6g}</td><td>{case['full_box']['damage']:.6g}</td><td>{case['full_box']['flipped']}</td></tr>
              <tr><td>Foreground intersection</td><td>{case['foreground_pixels']}</td><td>{case['foreground']['activation']:.6g}</td><td>{case['foreground']['damage']:.6g}</td><td>{case['foreground']['flipped']}</td></tr>
              <tr><td>Box complement</td><td>{case['complement_pixels']}</td><td>{case['complement']['activation']:.6g}</td><td>{case['complement']['damage']:.6g}</td><td>{case['complement']['flipped']}</td></tr>
            </tbody></table></article>'''
        )

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Foreground versus box-complement damage</title>
    <style>body{{margin:0;background:#f3f5f7;color:#1f2937;font:15px system-ui}}main{{max-width:1450px;margin:24px auto;padding:0 20px}}h1{{font-size:28px}}p{{line-height:1.5}}.note{{background:#fff7d6;border-left:5px solid #d99a00;padding:12px 16px}}article{{background:white;border-radius:12px;padding:20px;margin:22px 0}}.images{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}figure{{margin:0}}img{{display:block;width:100%}}figcaption{{padding:7px 0;font-weight:600}}table{{border-collapse:collapse;width:100%;margin:14px 0}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #dce2e8}}.filters{{position:sticky;top:0;background:#f3f5f7;padding:10px 0}}select{{font:inherit;padding:7px}}[hidden]{{display:none!important}}@media(max-width:850px){{.images{{grid-template-columns:1fr 1fr}}}}</style></head>
    <body><main><h1>BiRefNet foreground intersection versus box complement</h1>
    <p>Same 30 images and 57 predicates as the fresh blur-only ResNet demo. Each terminal Score-CAM box is partitioned exactly into the saved BiRefNet foreground intersection and the remaining pixels inside the box. Every intervention uses radius-20 blur.</p>
    <p class="note"><b>Important:</b> the complement is not necessarily pure context. It may contain background, object boundaries, thin structures, shadows, or object pixels missed by BiRefNet. Damage is the decrease in signed threshold margin, so positive values suppress either positive or negative predicates consistently.</p>
    <table><thead><tr><th>Population</th><th>N</th><th>Foreground larger</th><th>Complement larger</th><th>Tie</th><th>Foreground flips</th><th>Complement flips</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
    <p>Within the {acc_s['tested']} validated full boxes: foreground only flips {acc_s['foreground_only_flipped']}, complement only flips {acc_s['complement_only_flipped']}, both flip {acc_s['both_partition_members_flipped']}, and neither alone flips {acc_s['neither_partition_member_flipped']}.</p>
    <div class="filters"><label>Show <select id="filter"><option value="all">all terminal boxes</option><option value="valid">validated full boxes</option><option value="foreground">foreground larger damage</option><option value="complement">complement larger damage</option></select></label></div>
    {''.join(cards)}
    <p><a href="results.json">Full numerical results and provenance</a> · <a href="source_results.json">Preserved source results</a></p>
    </main><script>const f=document.getElementById('filter');f.onchange=()=>document.querySelectorAll('article').forEach(a=>{{a.hidden=f.value==='valid'?a.dataset.valid!=='true':f.value==='foreground'?a.dataset.winner!=='foreground':f.value==='complement'?a.dataset.winner!=='complement':false}})</script></body></html>'''
    (OUT / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    shutil.copy2(__file__, OUT / "executed_source.py")
    shutil.copy2(SOURCE / "results.json", OUT / "source_results.json")
    shutil.copy2(SOURCE / "sampling.json", OUT / "sampling.json")

    source = json.loads((SOURCE / "results.json").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = FeatureModel("resnet", WEIGHTS, "cuda")
    data = {
        "metadata": {
            "source_results_sha256": digest(SOURCE / "results.json"),
            "script_sha256": digest(__file__),
            "source_trial": str(SOURCE),
            "selection": "All 57 predicates from all 30 fixed images; no outcome filtering",
            "partition": "foreground = tested_box_union AND saved BiRefNet display_only; complement = tested_box_union AND NOT foreground",
            "intervention": {"kind": "PIL GaussianBlur pasted only within tested region", "radius": BLUR_RADIUS},
            "damage": "original signed threshold margin minus post-ablation signed threshold margin",
            "checks": {
                "partition_disjoint": True,
                "partition_union_equals_box": True,
                "outside_region_unchanged": True,
                "full_box_activation_reproduced": True,
            },
        },
        "cases": [],
    }

    for image_index, entry in enumerate(source["images"], 1):
        image_folder = SOURCE / entry["key"]
        image = Image.open(image_folder / "original.png").convert("RGB")
        rgb = np.asarray(image)
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)))
        z, predicted = model.evaluate(image)
        assert predicted == entry["predicted_class"]

        for predicate in entry["predicates"]:
            physical_id = int(predicate["physical_id"])
            branch = predicate["branch"]
            threshold = float(predicate["threshold"])
            original_activation = float(z[physical_id])
            np.testing.assert_allclose(original_activation, predicate["original_activation"], rtol=1e-5, atol=1e-6)
            original_margin = signed_margin(branch, threshold, original_activation)
            assert original_margin >= -1e-7
            terminal = predicate["attempts"][-1]
            arrays = np.load(image_folder / f"p{predicate['logical_id']}" / "final_masks.npz")
            box = arrays["tested_box_union"].astype(bool)
            foreground = arrays["display_only"].astype(bool)
            assert np.all(~foreground | box)
            complement = box & ~foreground
            assert not np.any(foreground & complement)
            assert np.array_equal(foreground | complement, box)
            assert int(box.sum()) == int(terminal["pixels"])

            key = f"{entry['key']}_p{predicate['logical_id']}"
            folder = OUT / key
            folder.mkdir()
            Image.fromarray(outline(rgb, foreground, complement)).save(folder / "regions.png")
            np.savez_compressed(folder / "masks.npz", full_box=box, foreground=foreground, complement=complement)
            record = {
                "key": key,
                "image_key": entry["key"],
                "image_name": entry["image_name"],
                "source_accepted": bool(predicate["accepted"]),
                "cutoff": float(terminal["cutoff"]),
                "box_pixels": int(box.sum()),
                "foreground_pixels": int(foreground.sum()),
                "complement_pixels": int(complement.sum()),
                "original_activation": original_activation,
                "original_margin": original_margin,
                "predicate": {
                    field: predicate[field]
                    for field in ("class_id", "physical_id", "logical_id", "branch", "threshold", "rank")
                },
            }
            for name, mask in (("full_box", box), ("foreground", foreground), ("complement", complement)):
                edited = compose_edit(rgb, blurred, mask)
                assert np.array_equal(edited[~mask], rgb[~mask])
                values, edited_class = model.evaluate(Image.fromarray(edited))
                value = float(values[physical_id])
                edited_margin = signed_margin(branch, threshold, value)
                record[name] = {
                    "activation": value,
                    "margin": edited_margin,
                    "damage": damage(original_margin, edited_margin),
                    "flipped": not is_active(branch, threshold, value),
                    "edited_predicted_class": edited_class,
                }
                if name == "full_box":
                    np.testing.assert_allclose(value, terminal["activation"], rtol=1e-5, atol=1e-6)
                Image.fromarray(edited).save(folder / f"{name}.png")

            difference = record["foreground"]["damage"] - record["complement"]["damage"]
            record["comparison"] = "foreground" if difference > TOLERANCE else "complement" if difference < -TOLERANCE else "tie"
            record["nonadditive_interaction"] = (
                record["full_box"]["damage"]
                - record["foreground"]["damage"]
                - record["complement"]["damage"]
            )
            data["cases"].append(record)
        print(f"{image_index:02d}/30 {entry['image_name']}: {len(entry['predicates'])} predicates", flush=True)

    accepted = [case for case in data["cases"] if case["source_accepted"]]
    data["summary"] = {
        "all_terminal_boxes": subset_summary(data["cases"]),
        "validated_full_boxes": subset_summary(accepted),
    }
    (OUT / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    render(data)
    print(json.dumps(data["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
