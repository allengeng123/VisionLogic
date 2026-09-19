"""Fixed descriptive grounding evaluation. Does not modify the manuscript.

Uses existing neuron Score-CAM, initial pathways, frozen thresholds and box
geometry. The only new algorithm is the prespecified area-matched control.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageFilter

from core import FeatureModel, compose_edit, digest, geometry, load_predicates, original_prefix
from scorecam_all_active import masked_scores, map_for_predicate
from compare_neuron_cams import LAYERS
from render_heatmap_components import components

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUT = ROOT / 'results/grounding_random100_four_models_v1'
DATASET = Path(os.environ.get('VISIONLOGIC_IMAGENET_VAL', WORKSPACE / 'data/imagenet-val'))
WEIGHTS = Path(os.environ.get('VISIONLOGIC_WEIGHTS', WORKSPACE / 'data/weights'))
PREDICATES = Path(os.environ.get('VISIONLOGIC_PREDICATES', WORKSPACE / 'results/section4_1/by_model'))
MODELS = ('resnet', 'vit', 'convnext', 'swin')
CUTOFFS = [round(i / 100, 2) for i in range(60, 14, -5)]
SEED = 42
PIXELS = 224 * 224


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def union_mask(boxes, shape=(224, 224)):
    mask = np.zeros(shape, dtype=bool)
    for b in boxes:
        assert 0 <= b['x0'] < b['x1'] <= shape[1]
        assert 0 <= b['y0'] < b['y1'] <= shape[0]
        mask[b['y0']:b['y1'], b['x0']:b['x1']] = True
    return mask


def matched_dimensions(area, aspect, width=224, height=224):
    """Match area near the bounded ideal aspect ratio, without response access.

    Clip the ideal real-valued width to the feasible range [area/H, W].
    Test integer widths/heights within one pixel of floor/ceil of that ideal,
    pairing each with floor/ceil of area divided by the chosen dimension.
    Prioritize absolute pixel-area error, then log-aspect error, then width.
    """
    assert 0 < area <= width * height and aspect > 0
    ideal_w = min(max(math.sqrt(area * aspect), area / height, 1.0), width, area)
    ideal_h = area / ideal_w
    choices = set()
    for w in range(max(1, math.floor(ideal_w) - 1), min(width, math.ceil(ideal_w) + 1) + 1):
        for h in {math.floor(area / w), math.ceil(area / w)}:
            if 1 <= h <= height:
                choices.add((w, h))
    for h in range(max(1, math.floor(ideal_h) - 1), min(height, math.ceil(ideal_h) + 1) + 1):
        for w in {math.floor(area / h), math.ceil(area / h)}:
            if 1 <= w <= width:
                choices.add((w, h))
    assert choices
    return min(choices, key=lambda wh: (abs(wh[0] * wh[1] - area), abs(math.log((wh[0] / wh[1]) / aspect)), wh))


def proposal_schedule(heat, seed):
    """Complete guided/control geometries before either search observes outputs."""
    assert heat.shape == (224, 224) and np.isfinite(heat).all() and (heat >= 0).all()
    rng = np.random.default_rng(seed)
    guided, control = [], []
    for cutoff in CUTOFFS:
        selected, boxes, peak = components(heat, cutoff)
        area = int(union_mask(boxes).sum())
        assert not np.any(selected & ~union_mask(boxes))
        guided.append(dict(cutoff=cutoff, boxes=boxes, pixels=area))
        if area:
            bw = max(b['x1'] for b in boxes) - min(b['x0'] for b in boxes)
            bh = max(b['y1'] for b in boxes) - min(b['y0'] for b in boxes)
            w, h = matched_dimensions(area, bw / bh)
            x = int(rng.integers(0, 224 - w + 1))
            y = int(rng.integers(0, 224 - h + 1))
            random_boxes = [dict(x0=x, y0=y, x1=x+w, y1=y+h)]
        else:
            random_boxes, w, h, bw, bh = [], 0, 0, 0, 0
        control.append(dict(cutoff=cutoff, boxes=random_boxes, pixels=w*h,
                            matched_guided_pixels=area, area_error_pixels=w*h-area,
                            guided_enclosing_aspect=bw/bh if bh else None))
    return guided, control


def run_search(model, predicate, rgb, blur, schedule):
    attempts = []
    for proposal in schedule:
        mask = union_mask(proposal['boxes'])
        assert int(mask.sum()) == proposal['pixels']
        attempt = dict(cutoff=proposal['cutoff'], pixels=proposal['pixels'],
                       activation=None, flipped=False, evaluated=False)
        if mask.any():
            edited = compose_edit(rgb, blur, mask)
            assert np.array_equal(edited[~mask], rgb[~mask])
            values, prediction = model.evaluate(Image.fromarray(edited))
            assert np.isfinite(values).all()
            value = float(values[predicate.physical_id])
            attempt.update(activation=value, flipped=not predicate.active(value),
                           evaluated=True, edited_prediction=prediction,
                           signed_margin=predicate.margin(value),
                           edited_rgb_sha256=hashlib.sha256(edited.tobytes()).hexdigest())
        attempts.append(attempt)
        if attempt['flipped']:
            break
    passed = attempts[-1]['flipped']
    return dict(success=passed, accepted_step=len(attempts)-1 if passed else None,
                accepted_area_pixels=attempts[-1]['pixels'] if passed else None,
                attempts=attempts, model_evaluations=sum(a['evaluated'] for a in attempts))


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'sample.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['seed'] == SEED and len(manifest['images']) == 100
        return manifest
    files = sorted(p for p in DATASET.glob('*/*') if p.is_file() and re.fullmatch(r'ILSVRC2012_val_\d{8}\.JPEG', p.name))
    by_id = {int(p.stem.rsplit('_', 1)[1]): p for p in files}
    assert len(by_id) == len(files), 'Duplicate original validation IDs'
    assert set(by_id).issubset(set(range(1, 50001)))
    sampled_ids = random.Random(SEED).sample(range(1, 50001), 100)
    missing = [i for i in sampled_ids if i not in by_id]
    assert not missing, f'Sampled originals missing, no substitutions permitted: {missing}'
    picked = [by_id[i] for i in sampled_ids]
    images = [dict(index=i, image_id=p.stem, relative_path=p.relative_to(DATASET).as_posix(),
                   synset=p.parent.name, sha256=digest(p))
              for i, p in enumerate(picked)]
    manifest = dict(seed=SEED, sampler='Python random.Random(42).sample(range(1,50001),100), without replacement; no resampling',
                    dataset_root=str(DATASET), population_images=50000, population_classes=1000,
                    local_original_images=len(files), local_original_classes=len({p.parent.name for p in files}),
                    local_missing_ids=sorted(set(range(1,50001))-set(by_id)), sampled_missing_ids=missing,
                    input_audit='Only canonical ILSVRC2012_val_XXXXXXXX.JPEG files; edited PNG excluded; sample drawn from full canonical ID population, not available files',
                    local_listing_sha256=hashlib.sha256('\n'.join(p.relative_to(DATASET).as_posix() for p in files).encode()).hexdigest(),
                    images=images)
    write_json(manifest_path, manifest)
    return manifest


def provenance():
    sources = ['evaluate_grounding_random100.py', 'core.py', 'scorecam_all_active.py',
               'compare_neuron_cams.py', 'render_heatmap_components.py', 'render_explanations.py']
    directory = OUT / 'source'
    directory.mkdir(exist_ok=True)
    for name in sources:
        dst = directory / name
        if dst.exists():
            assert digest(dst) == digest(ROOT / name), f'Source changed during evaluation: {name}'
        else:
            shutil.copy2(ROOT / name, dst)
    configuration = dict(models=MODELS, cutoffs=CUTOFFS, blur_radius=20, sampling_seed=42,
                         control_seed=42, control_rng='NumPy PCG64; one independently seeded search per target',
                         target_seed='first 16 hex digits of SHA256(42|model|image_id|predicted_class|logical_id)',
                         preprocessing='Resize short side to 256 bilinear, center crop 224; ImageNet normalization; same crop for all models',
                         targets='All and only J_c(x): initial-pathway signed selections in frozen predicted-class vocabulary and active at original input',
                         threshold_source='section42-results-no-extension/results/*/predicate_metrics.csv.gz, train rows only',
                         acceptance='First nonempty box proposal whose radius-20 blur deactivates fixed predicate; no segmentation or retention test',
                         zero_map_policy='All ten proposals empty, failure for both methods; target remains in denominator',
                         control='One uniformly positioned rectangle per nonempty guided area step; no overlap avoidance; no outcome-based geometry',
                         dimensions=matched_dimensions.__doc__, target_layers=LAYERS,
                         scorecam_batch_size=8, scorecam_channels='all',
                         sorting=dict(vit='stable', resnet='stable', convnext='quicksort', swin='quicksort'),
                         source_sha256={n:digest(ROOT/n) for n in sources},
                         python=platform.python_version(), device=torch.cuda.get_device_name(0),
                         versions={n:importlib.metadata.version(n) for n in ['torch', 'torchvision', 'numpy', 'Pillow', 'grad-cam', 'opencv-python']})
    config_path = OUT / 'configuration.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(configuration))
    else:
        write_json(config_path, configuration)
    paper_path = OUT / 'manuscript_hashes_before.json'
    if not paper_path.exists():
        paper = WORKSPACE / 'iclr2027_submission/VisionLogic_Working'
        paths = [p for p in paper.rglob('*') if p.is_file()]
        paths += [WORKSPACE/'output/pdf/VisionLogic_ScoreCAM_BoxOnly_Grounding.pdf', WORKSPACE/'output/VisionLogic_ScoreCAM_BoxOnly_Grounding_Overleaf.zip']
        write_json(paper_path, {str(p.relative_to(WORKSPACE)):digest(p) for p in paths})


def run_model(name, manifest, limit=None):
    dest = OUT / name
    dest.mkdir(exist_ok=True)
    path = PREDICATES / name / 'predicate_metrics.csv.gz'
    vocab = load_predicates(path)
    classes = {c for c, _ in vocab}
    model = FeatureModel(name, WEIGHTS, 'cuda')
    weights, bias = model.weight.cpu().numpy(), model.bias.cpu().numpy()
    write_json(dest/'provenance.json', dict(weights_sha256=digest(model.weight_path), thresholds_sha256=digest(path),
                                           head_reconstruction_max_error=model.head_reconstruction_max_error))
    for item in manifest['images'][:limit]:
        key = item['image_id']
        result_path = dest / f'{key}.json'
        if result_path.exists():
            continue
        start = time.perf_counter()
        image_path = DATASET / item['relative_path']
        assert digest(image_path) == item['sha256']
        folder = dest / key
        folder.mkdir(exist_ok=True)
        image = geometry('resize256_bilinear')(Image.open(image_path).convert('RGB'))
        image.save(folder/'input.png')
        rgb = np.asarray(image)
        blur = np.asarray(image.filter(ImageFilter.GaussianBlur(20)))
        z, prediction = model.evaluate(image)
        assert np.isfinite(z).all()
        prefix = original_prefix(z, prediction, weights, bias, name)
        partial = bias.copy()
        for i, lid in enumerate(prefix):
            j = lid % len(z)
            partial += weights[:, j] * z[j]
            if i < len(prefix)-1:
                assert int(partial.argmax()) != prediction
        assert int(partial.argmax()) == prediction, 'Initial pathway failed prediction reconstruction'
        np.savez_compressed(folder/'original_activations.npz', z=z, pathway=np.array(prefix), partial_logits=partial)
        active = [p.logical_id for (c, _), p in vocab.items() if c == prediction and p.active(z[p.physical_id])]
        eligible = [vocab[(prediction, lid)] for lid in prefix if (prediction, lid) in vocab and lid in active]
        record = dict(model=name, **item, predicted_class=prediction, vocabulary_available=prediction in classes,
                      pathway=prefix, active_predicate_ids=active, eligible_predicate_ids=[p.logical_id for p in eligible],
                      selected_without_vocabulary=[lid for lid in prefix if (prediction, lid) not in vocab],
                      selected_but_inactive=[lid for lid in prefix if (prediction, lid) in vocab and lid not in active],
                      input_sha256=digest(folder/'input.png'), predicates=[])
        if eligible:
            check, native, scores = masked_scores(model, image)
            np.testing.assert_allclose(check, z, rtol=1e-5, atol=1e-6)
            assert np.isfinite(native).all() and np.isfinite(scores).all()
            # Save exact heatmaps; the shared forwards are identical for all targets.
            maps = {}
            for p in eligible:
                heat, _ = map_for_predicate(native, scores, p)
                maps[f'p{p.logical_id}'] = heat
            np.savez_compressed(folder/'heatmaps.npz', **maps)
            for p in eligible:
                heat = maps[f'p{p.logical_id}']
                seed_text = f'{SEED}|{name}|{key}|{prediction}|{p.logical_id}'
                target_seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                guided, control = proposal_schedule(heat, target_seed)
                # Both complete schedules exist before any ablation forward.
                geometry_record = dict(guided=guided, random=control, control_seed=str(target_seed))
                write_json(folder/f'p{p.logical_id}_geometry.json', geometry_record)
                result = dict(predicate=asdict(p), original_activation=float(z[p.physical_id]),
                              zero_heatmap=bool(heat.max() == 0), control_seed=str(target_seed),
                              guided=run_search(model, p, rgb, blur, guided),
                              random=run_search(model, p, rgb, blur, control))
                record['predicates'].append(result)
            del native, scores, maps
        record['seconds'] = time.perf_counter()-start
        write_json(result_path, record)
        n = len(record['predicates'])
        g = sum(p['guided']['success'] for p in record['predicates'])
        r = sum(p['random']['success'] for p in record['predicates'])
        print(f'{name} {item["index"]+1}/100 {key}: eligible={n} guided={g} random={r} seconds={record["seconds"]:.1f}', flush=True)
        write_json(OUT/'progress.json', dict(model=name, last_image_index=item['index'], last_image=key, time=time.time()))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Diagnostic prefix; resume completes original sample, no resampling')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = initialize()
    provenance()
    if args.prepare_only:
        print(f'Prepared {len(manifest["images"])} of {manifest["population_images"]} images; no model filtering.', flush=True)
        return
    try:
        for name in args.models:
            run_model(name, manifest, args.limit)
    except Exception:
        (OUT/'technical_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    print('REQUESTED RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
