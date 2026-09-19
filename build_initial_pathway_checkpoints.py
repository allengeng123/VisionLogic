"""Extract prediction-preserving source pathways without signature extension.

Historical extended checkpoints are optional and, when supplied, are used only
to audit that the extracted initial pathway is a stored prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from visionlogic_section42_predicate_analysis import (
    MODEL_HEADS, activation_files, extract_path, load_matrix, source_index,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build(model: str, train_dir: Path, weights_path: Path,
          output_path: Path, extended_path: Path | None = None) -> dict:
    started = time.time()
    weight_key, bias_key, sort_kind = MODEL_HEADS[model]
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    weights = state[weight_key].detach().cpu().numpy()
    bias = state[bias_key].detach().cpu().numpy()
    dimension = int(weights.shape[1])
    files = activation_files(train_dir)
    sources, raw_rows, pre_overwrite_rows = source_index(files, weights, bias)
    extended = None
    if extended_path is not None:
        with extended_path.open('rb') as handle:
            extended_payload = pickle.load(handle)
        extended = {int(k): v for k, v in extended_payload['resolved_pruned_dataset'].items()}
        if set(sources) != set(extended):
            raise RuntimeError(f'{model}: source/checkpoint class keys differ')

    initial = {}
    images = 0
    extension_lengths = []
    for class_number, class_id in enumerate(sorted(sources), 1):
        source = sources[class_id]
        matrix = load_matrix(source['path'])[source['indices']]
        stored = extended[class_id] if extended is not None else [None] * len(matrix)
        if len(matrix) != len(stored):
            raise RuntimeError(f'{model} class {class_id}: row count mismatch')
        paths = []
        for row, extended_pathway in zip(matrix, stored):
            path = np.asarray(
                extract_path(row, class_id, weights, bias, dimension, sort_kind),
                dtype=np.int64,
            )
            if extended_pathway is not None:
                extended_array = np.asarray(extended_pathway, dtype=np.int64)
                if len(path) > len(extended_array) or not np.array_equal(
                        path, extended_array[:len(path)]):
                    raise RuntimeError(f'{model} class {class_id}: stored path is not an initial-path prefix')
                extension_lengths.append(int(len(extended_array) - len(path)))
            paths.append(path)
        initial[class_id] = paths
        images += len(paths)
        if class_number % 100 == 0:
            print(f'{model}: reconstructed {class_number}/{len(sources)} classes', flush=True)

    extension_lengths_array = np.asarray(extension_lengths, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Compatibility key used by the existing analysis script. These values
        # are initial pathways and have not undergone conflict resolution.
        'resolved_pruned_dataset': initial,
        'initial_pathways': initial,
        'method': 'initial_prediction_preserving_pathways_without_extension',
        'source_extended_checkpoint_sha256': sha256(extended_path) if extended_path else None,
        'N_physical': dimension,
        'extension_applied': False,
    }
    with output_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    unique, counts = np.unique(extension_lengths_array, return_counts=True)
    return {
        'model': model,
        'classes': len(initial),
        'images': images,
        'dimension': dimension,
        'raw_source_rows': raw_rows,
        'retained_rows_before_overwrite': pre_overwrite_rows,
        'extension_length_distribution_in_historical_checkpoint': {
            str(int(k)): int(v) for k, v in zip(unique, counts)
        },
        'historical_extended_images': int(np.sum(extension_lengths_array > 0)) if extended is not None else None,
        'historical_max_extension': int(extension_lengths_array.max(initial=0)) if extended is not None else None,
        'all_initial_paths_verified_as_stored_prefixes': True if extended is not None else None,
        'weights_sha256': sha256(weights_path),
        'train_directory': str(train_dir),
        'historical_checkpoint': str(extended_path) if extended_path else None,
        'output_checkpoint': str(output_path),
        'output_sha256': sha256(output_path),
        'seconds': time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('initial-pathway-checkpoints'))
    parser.add_argument('--weights-dir', type=Path, required=True,
                        help='Directory containing torchvision model weight files.')
    parser.add_argument('--activation-root', type=Path, required=True,
                        help='Directory with vit/resnet/convnext/swin train subdirectories.')
    parser.add_argument('--extended-checkpoint-dir', type=Path,
                        help='Optional historical checkpoint directory for prefix verification.')
    args = parser.parse_args()
    cache = args.weights_dir
    activations = args.activation_root
    checkpoints = args.extended_checkpoint_dir
    specs = {
        'vit': (
            activations/'vit/train', cache/'vit_b_16-c867db91.pth',
            checkpoints/'phase3_lookahead_vit.pkl' if checkpoints else None),
        'resnet': (
            activations/'resnet/train', cache/'resnet50-11ad3fa6.pth',
            checkpoints/'phase3_lookahead_resnet.pkl' if checkpoints else None),
        'convnext': (
            activations/'convnext/train', cache/'convnext_base-6075fbad.pth',
            checkpoints/'phase3_lookahead_convnext.pkl' if checkpoints else None),
        'swin': (
            activations/'swin/train', cache/'swin_t-704ceda3.pth',
            checkpoints/'phase3_lookahead_swin.pkl' if checkpoints else None),
    }
    reports = []
    for model, (train, weights, extended) in specs.items():
        output = args.output_dir/f'phase3_initial_{model}.pkl'
        reports.append(build(model, train, weights, output, extended))
        (args.output_dir/'manifest.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(json.dumps(reports[-1], indent=2), flush=True)


if __name__ == '__main__':
    main()
