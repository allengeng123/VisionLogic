"""Small, fully logged four-model grounding demonstration (not a benchmark)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import html
import json
import os
from pathlib import Path
import platform
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter

from core import (FeatureModel, ISNet, Lama, SPECS, compose_edit, digest, geometry,
                  load_predicates, mask_from_mass, neuron_heatmap, original_prefix)

ROOT = Path(__file__).resolve().parent
POLICIES = dict(vit='resize256_bilinear', resnet='resize256_bilinear',
                convnext='square232_bilinear', swin='square232_bicubic')
DEFAULTS = dict(heat_mass_schedule=[0.60, 0.75, 0.90], max_mask_fraction=0.50,
                ig_steps=32, ig_max_steps=512, ig_completeness_relative_tolerance=0.05,
                ig_smoothing_sigma_pixels=2, mask_closing_kernel_pixels=5, isnet_threshold=0.5,
                isnet_boundary_band_pixels=3, blur_radius_pixels=10)

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

def components(mask):
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]-1)

def run_removal(model, image, predicate, heat, foreground, lama, out, settings):
    """At most 3 nested attempts; stop on first flip, never search for minimum."""
    attempts = []
    previous = np.zeros(heat.shape, dtype=bool)
    original = np.array(image)
    final_mask, final_edit = previous, image
    status = 'unvalidated'
    for attempt, mass in enumerate(settings['heat_mass_schedule']):
        mask = mask_from_mass(heat, mass, foreground) | previous
        area = float(mask.mean())
        record = dict(attempt=attempt, heat_mass=mass, area_fraction=area,
                      components=components(mask))
        if not mask.any():
            attempts.append(dict(**record, status='empty_support_map'))
            break
        if area > settings['max_mask_fraction']:
            attempts.append(dict(**record, status='area_guardrail'))
            break
        if attempt and np.array_equal(mask, previous):
            attempts.append(dict(**record, status='duplicate_mask'))
            continue
        previous = mask
        edited = lama(image, mask)
        values, edited_class = model.evaluate(edited)
        value = float(values[predicate.physical_id])
        active = predicate.active(value)
        # Blur is logged ONLY as a secondary intervention, never a fallback pass.
        blurred = np.array(image.filter(ImageFilter.GaussianBlur(settings['blur_radius_pixels'])))
        blur_image = Image.fromarray(compose_edit(original, blurred, mask))
        blur_values, _ = model.evaluate(blur_image)
        blur_value = float(blur_values[predicate.physical_id])
        edited.save(out / f'attempt_{attempt}_lama.png')
        blur_image.save(out / f'attempt_{attempt}_blur.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(out / f'attempt_{attempt}_mask.png')
        outside_equal = bool(np.array_equal(np.array(edited)[~mask], original[~mask]))
        if not outside_equal:
            raise AssertionError('Inpainting changed outside-mask pixels')
        record.update(status='predicate_flipped' if not active else 'predicate_still_active',
                      activation=value, margin=predicate.margin(value), predicate_active=active,
                      edited_model_class=edited_class, outside_mask_identical=outside_equal,
                      blur_activation=blur_value, blur_predicate_active=predicate.active(blur_value))
        attempts.append(record)
        final_mask, final_edit = mask, edited
        if not active:
            status = 'single_fill_flip'
            break
    return attempts, status, final_mask, final_edit

def draw_case(path, image, heat, mask, edited, title, edited_was_run=True):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    axes[0].imshow(image)
    axes[1].imshow(image)
    if heat.max() > 0:
        axes[1].imshow(heat, cmap='magma', alpha=0.65, vmin=0, vmax=float(heat.max()))
    axes[2].imshow(image)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [0, 0.85, 0.9, 0.5]
    axes[2].imshow(overlay)
    axes[3].imshow(edited)
    for ax, label in zip(axes, ['Model input crop', 'Signed neuron IG', 'Joint removal mask',
                              'LaMa replacement' if edited_was_run else 'No replacement performed']):
        ax.set_title(label, fontsize=11)
        ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(path, dpi=140, facecolor='white')
    plt.close(fig)

def write_report(output, records, metadata):
    summaries = {}
    for name in SPECS:
        rows = [r for r in records if r['model'] == name]
        attempted = [r for r in rows if 'attempts' in r]
        evaluations = [a for r in attempted for a in r['attempts'] if 'activation' in a]
        summaries[name] = dict(images=len(rows), eligible_predicate_cases=len(attempted),
                               single_fill_flips=sum(r['status']=='single_fill_flip' for r in attempted),
                               image_level_skips=len(rows)-len(attempted),
                               total_inpainting_attempts=len(evaluations))
    dump(output/'results.json', dict(metadata=metadata, summary=summaries, cases=records))
    pieces = ['<!doctype html><html><head><meta charset="utf-8"><title>VisionLogic grounding demo</title>',
              '<style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 24px;color:#172337}',
              'img{width:100%;border:1px solid #ddd}pre{white-space:pre-wrap}section{margin:30px 0}',
              'table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}</style></head><body>',
              '<h1>Experimental predicate grounding — four-model demo</h1>',
              '<p>Frozen training thresholds; exact head-input neurons; no rule prediction or logit attribution. '
              'Preselected validation images and a fixed eligible original-prefix rank per image. '
              'Failures are retained. These counts are not ImageNet-wide performance.</p>',
              '<p><strong>Single deterministic LaMa fill:</strong> a threshold flip is an operational result, '
              'not yet a claim of realistic cue removal or robustness across replacement samples. '
              'All coordinates refer to the displayed model input crop.</p>',
              '<table><tr><th>Model</th><th>Images</th><th>Eligible cases</th><th>Single-fill flips</th><th>Edits</th></tr>']
    for name, s in summaries.items():
        pieces.append(f"<tr><td>{name}</td><td>{s['images']}</td><td>{s['eligible_predicate_cases']}</td>"
                      f"<td>{s['single_fill_flips']}</td><td>{s['total_inpainting_attempts']}</td></tr>")
    pieces.append('</table>')
    for r in records:
        pieces.append('<section><h2>'+html.escape(r['model']+' / '+r['synset']+' / '+r['status'])+'</h2>')
        if 'figure' in r:
            pieces.append('<img src="'+html.escape(r['figure'])+'">')
        pieces.append('<details><summary>Exact predicate and all attempts</summary><pre>'+html.escape(json.dumps(r,indent=2))+'</pre></details></section>')
    pieces.append('<h2>Execution settings and provenance</h2><pre>'+html.escape(json.dumps(metadata,indent=2))+'</pre></body></html>')
    (output/'index.html').write_text('\n'.join(pieces), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--weights', type=Path,
                        default=Path(os.environ.get('VISIONLOGIC_WEIGHTS', ROOT.parent/'data/weights')))
    parser.add_argument('--source-metrics', type=Path, default=ROOT.parent/'section42-results/results')
    parser.add_argument('--images', type=Path, default=ROOT/'preflight/images.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/demo_v1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--predicate-rank', type=int, default=0,
                        help='Fixed zero-based rank in eligible original prefix; never chosen from edit outcomes')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Refuse accidental overwrites of experiments.
    args.output.mkdir(parents=True, exist_ok=False)
    images = json.loads(args.images.read_text())
    if args.predicate_rank < 0:
        parser.error('--predicate-rank must be nonnegative')
    metadata = dict(version='experimental-grounding-v2', settings=DEFAULTS, preprocessing=POLICIES,
                    python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                    numpy=np.__version__, device=args.device, image_manifest_sha256=digest(args.images),
                    attribution_target='signed physical classifier-input neuron; class fixed before perturbation',
                    thresholds='unchanged train rows from Section 4.2 predicate_metrics.csv.gz',
                    predicate_selection=f'eligible original-prefix rank {args.predicate_rank}; active source-vocabulary members only',
                    inference_uses_original_prediction='yes, ONLY to choose original class vocabulary/prefix; then frozen',
                    inpainting='deterministic big-LaMa, one fill per mask; no sampling robustness claim',
                    mask_acceptance='first LaMa predicate flip within 3 attempts and 50% input-area guardrail',
                    source_metrics_sha256={}, classifier_weight_sha256={},
                    code_sha256={n:digest(ROOT/n) for n in ['core.py','run_demo.py']},
                    assets=json.loads((ROOT/'assets/manifest.json').read_text()))
    records = []
    segmentation_cache = {}
    segmenter = ISNet(ROOT/'assets/isnet-general-use.onnx')
    lama = Lama(ROOT/'assets/big-lama.pt', args.device)
    for name in args.models:
        model = FeatureModel(name, args.weights, args.device)
        metadata['classifier_weight_sha256'][name] = digest(model.weight_path)
        source = args.source_metrics/name/'predicate_metrics.csv.gz'
        predicates = load_predicates(source)
        metadata['source_metrics_sha256'][name] = digest(source)
        weights, bias = model.weight.cpu().numpy(), model.bias.cpu().numpy()
        for item in images:
            started = time.time()
            image = geometry(POLICIES[name])(Image.open(item['path']).convert('RGB'))
            z, predicted = model.evaluate(image)
            prefix = original_prefix(z, predicted, weights, bias, name)
            eligible = [predicates[(predicted, lid)] for lid in prefix
                        if (predicted, lid) in predicates and predicates[(predicted,lid)].active(z[lid % model.dimension])]
            record = dict(model=name, **item, image_sha256=digest(item['path']), predicted_class=predicted,
                          preprocessing=POLICIES[name], original_prefix=prefix,
                          eligible_logical_ids=[p.logical_id for p in eligible],
                          head_reconstruction_max_error=model.head_reconstruction_max_error)
            if len(eligible) <= args.predicate_rank:
                record['status'] = 'no_eligible_predicate_at_requested_rank'
                records.append(record)
                write_report(args.output, records, metadata)
                continue
            predicate = eligible[args.predicate_rank]
            key = name+'_'+item['synset']+'_'+Path(item['path']).stem
            out = args.output/key
            out.mkdir()
            image.save(out/'original.png')
            record.update(predicate=asdict(predicate), original_activation=float(z[predicate.physical_id]),
                          original_margin=predicate.margin(z[predicate.physical_id]))
            print('Attributing', name, item['synset'], record['predicate'], flush=True)
            heat, signed, attribution = neuron_heatmap(model, image, predicate, DEFAULTS['ig_steps'])
            cache_key = (item['path'], POLICIES[name])
            if cache_key not in segmentation_cache:
                segmentation_cache[cache_key] = segmenter(image)
            foreground = segmentation_cache[cache_key]
            np.savez_compressed(out/'maps.npz', heat=heat, signed_ig=signed, isnet=foreground)
            Image.fromarray((foreground*255).astype(np.uint8)).save(out/'isnet.png')
            if attribution['numerical_converged']:
                attempts, status, mask, edit = run_removal(model, image, predicate, heat, foreground, lama, out, DEFAULTS)
            else:
                attempts, status, mask, edit = [], 'attribution_not_converged', np.zeros(heat.shape,dtype=bool), image
            record.update(attribution=attribution, attempts=attempts, status=status,
                          final_mask_fraction=float(mask.mean()), final_components=components(mask),
                          elapsed_seconds=time.time()-started, figure=key+'/panel.png')
            evaluated = [a for a in attempts if 'activation' in a]
            after = evaluated[-1]['activation'] if evaluated else record['original_activation']
            operator = '>=' if predicate.branch == '+' else '<='
            title = (f"{name.upper()} | class {predicted}, z[{predicate.physical_id}] {operator} {predicate.threshold:.4f} | "
                     f"{record['original_activation']:.4f} -> {after:.4f} | {status} | mask {mask.mean():.1%}")
            draw_case(out/'panel.png', image, heat, mask, edit, title, bool(evaluated))
            dump(out/'case.json', record)
            records.append(record)
            write_report(args.output, records, metadata)
            print('RESULT', name, item['synset'], status, 'area',mask.mean(), 'seconds',round(time.time()-started,1), flush=True)
        del model
        gc.collect()
    print('Report:', args.output/'index.html', flush=True)

if __name__ == '__main__':
    main()
