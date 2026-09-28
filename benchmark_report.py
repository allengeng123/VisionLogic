"""Summarize the actual two-stage grounding outputs, including mirrored controls."""
import argparse
import json
from pathlib import Path
import statistics


def read_targets(folder):
    records = {}
    images = set()
    for path in sorted(folder.glob('ILSVRC2012_val_*.json')):
        row = json.loads(path.read_text())
        images.add(row['image_id'])
        for target in row['predicates']:
            key = (row['image_id'], target['predicate']['logical_id'])
            if key in records:
                raise ValueError(f'Duplicate target {key}')
            records[key] = target
    return images, records


def summarize(grad_folder, score_folder):
    images, grad = read_targets(grad_folder)
    score_images, score = read_targets(score_folder)
    if not images or images != score_images or grad.keys() != score.keys():
        raise ValueError('Both stages must contain the same nonempty image population and predicate keys')
    for key in grad:
        if grad[key]['predicate'] != score[key]['predicate'] or abs(grad[key]['original_activation'] - score[key]['original_activation']) > 1e-5:
            raise ValueError(f'Predicate or activation mismatch at {key}')
    result = dict(images=len(images), complete_100_images=len(images) == 100,
                  eligible_images=len({key[0] for key in grad}), targets=len(grad))
    for arm in ('guided', 'random'):
        wins, areas, evaluations = [], [], 0
        for key in grad:
            first = grad[key]['lower010'][arm]
            chosen = first if first['success'] else score[key]['lower010'][arm]
            evaluations += first['model_evaluations']
            if not first['success']:
                evaluations += chosen['model_evaluations']
            if chosen['success']:
                wins.append(key)
                areas.append(chosen['accepted_area_pixels'] / (224*224) * 100)
        result[arm] = dict(successes=len(wins), rate=len(wins)/len(grad) if grad else None,
                           images_with_success=len({key[0] for key in wins}),
                           median_area_percent=statistics.median(areas) if areas else None,
                           intervention_evaluations=evaluations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['vit','resnet','convnext','swin'], default=['vit','resnet'])
    parser.add_argument('--results', type=Path, default=Path('experimental_grounding/results'))
    parser.add_argument('--output', type=Path, default=Path('outputs/grounding-summary.json'))
    args = parser.parse_args()
    report = {model: summarize(args.results/'gradcam_random100_four_models_lower010_v1'/model,
                              args.results/'scorecam_random100_four_models_lower010_v1'/model)
              for model in args.models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
