"""Ground a single image with Grad-CAM followed by Score-CAM and blur removal."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from pytorch_grad_cam import GradCAM

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experimental_grounding'))
from core import FeatureModel, SPECS, digest, geometry, image_tensor, load_predicates, original_prefix, compose_edit
from compare_neuron_cams import LAYERS, SignedNeuronTarget, reshape_vit, reshape_swin
from scorecam_all_active import masked_scores, map_for_predicate
from render_heatmap_components import components
from evaluate_grounding_random100 import union_mask, run_search

CUTOFFS = [round(i / 100, 2) for i in range(60, 9, -5)]


def proposals(heat):
    if heat.shape != (224, 224) or not np.isfinite(heat).all() or (heat < 0).any():
        raise ValueError('Expected a finite, nonnegative 224 x 224 heatmap')
    result = []
    for cutoff in CUTOFFS:
        _, boxes, _ = components(heat, cutoff)
        result.append(dict(cutoff=cutoff, boxes=boxes, pixels=int(union_mask(boxes).sum())))
    return result


def ground(model, predicate, rgb, blur, grad_heat, score_heat):
    """The fallback is lazy; preserve failed attempts and stop at first success."""
    stages = []
    for method, make_heat in [('gradcam', lambda: grad_heat), ('scorecam', score_heat)]:
        heat = make_heat()
        schedule = proposals(heat)
        result = run_search(model, predicate, rgb, blur, schedule)
        stages.append(dict(method=method, **result))
        if result['success']:
            boxes = schedule[result['accepted_step']]['boxes']
            return stages, heat, boxes
    return stages, heat, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', choices=SPECS, default='resnet')
    parser.add_argument('--weights', type=Path, default=ROOT / 'data/weights')
    parser.add_argument('--predicates', type=Path, help='predicate_metrics.csv.gz from a matching protocol')
    parser.add_argument('--protocol', choices=['paper', 'legacy'], default='legacy',
                        help='Bundled thresholds use legacy; paper requires newly fitted predicates')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/explanation')
    args = parser.parse_args()
    if args.protocol == 'paper' and args.predicates is None:
        parser.error('--protocol paper requires --predicates fitted with the paper protocol; see docs/REPRODUCIBILITY.md')
    predicates_path = args.predicates or ROOT / f'results/section4_1/by_model/{args.model}/predicate_metrics.csv.gz'
    summary_path = predicates_path.with_name('summary.json')
    if args.protocol == 'paper' and not summary_path.is_file():
        parser.error('Paper predicates require their generated summary.json alongside the table')
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        protocol = summary.get('protocol', 'legacy')
        if protocol != args.protocol:
            parser.error('Predicate summary protocol does not match --protocol')
        if summary.get('model') != args.model:
            parser.error('Predicate summary model does not match --model')
    if not args.image.is_file() or not predicates_path.is_file():
        parser.error('Image and predicate table must exist')
    if args.output.exists():
        parser.error('Output already exists; choose a new --output directory')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.time()
    model = FeatureModel(args.model, args.weights, args.device)
    if summary_path.exists():
        expected_weights = summary.get('provenance', {}).get('weights_sha256')
        if expected_weights and expected_weights != digest(model.weight_path):
            parser.error('Predicate thresholds were fitted with different classifier weights')
    image = geometry('resize256_bilinear')(Image.open(args.image).convert('RGB'))
    rgb = np.asarray(image)
    blur = np.asarray(image.filter(ImageFilter.GaussianBlur(20)))
    z, predicted = model.evaluate(image)
    vocab = load_predicates(predicates_path)
    prefix = original_prefix(z, predicted, model.weight.cpu().numpy(), model.bias.cpu().numpy(), args.model, args.protocol)
    selected = [vocab[(predicted, lid)] for lid in prefix if (predicted, lid) in vocab and vocab[(predicted, lid)].active(z[lid % len(z)])]
    args.output.mkdir(parents=True)
    image.save(args.output / 'input.png')
    cached_score = None

    def score_heat(predicate):
        nonlocal cached_score
        if cached_score is None:
            checked_z, native, scores = masked_scores(model, image)
            np.testing.assert_allclose(z, checked_z, rtol=1e-5, atol=1e-6)
            cached_score = (native, scores)
        return map_for_predicate(*cached_score, predicate)[0]

    records = []
    for predicate in selected:
        reshape = {'vit': reshape_vit, 'swin': reshape_swin}.get(args.model)
        layer = model.net.get_submodule(LAYERS[args.model])
        with GradCAM(model=model.net, target_layers=[layer], reshape_transform=reshape) as cam:
            heat = cam(input_tensor=image_tensor(image, model.device).requires_grad_(True),
                       targets=[SignedNeuronTarget(predicate)])[0]
        stages, accepted_heat, boxes = ground(model, predicate, rgb, blur, heat, lambda: score_heat(predicate))
        stem = f'predicate_{predicate.logical_id}'
        np.save(args.output / f'{stem}_heatmap.npy', accepted_heat)
        mask = union_mask(boxes)
        Image.fromarray(mask.astype(np.uint8)*255).save(args.output / f'{stem}_mask.png')
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        for box in boxes:
            draw.rectangle((box['x0'], box['y0'], box['x1']-1, box['y1']-1), outline='#16a34a', width=2)
        panel.save(args.output / f'{stem}_boxes.png')
        if stages[-1]['success']:
            Image.fromarray(compose_edit(rgb, blur, mask)).save(args.output / f'{stem}_removed.png')
        records.append(dict(predicate=asdict(predicate), original_activation=float(z[predicate.physical_id]),
                            success=stages[-1]['success'], stages=stages, accepted_boxes=boxes))
        print(f'{stem}: {"accepted" if stages[-1]["success"] else "failed"}', flush=True)
    result = dict(model=args.model, protocol=args.protocol, predicted_class=predicted,
                  pathway=prefix, eligible_predicates=len(selected), predicates=records,
                  status='evaluated' if selected else 'no_active_selected_predicates',
                  weights_sha256=digest(model.weight_path), predicates_sha256=digest(predicates_path),
                  image_sha256=digest(args.image), cutoffs=CUTOFFS, blur_radius=20,
                  torch=torch.__version__, numpy=np.__version__, seconds=time.time()-started)
    (args.output / 'explanation.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Prediction: {predicted}; {len(selected)} eligible predicates; outputs: {args.output}')


if __name__ == '__main__':
    main()
