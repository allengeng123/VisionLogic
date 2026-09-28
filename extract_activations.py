"""Extract exact frozen-head inputs from class-organized ImageNet images."""
import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / 'experimental_grounding'))
from core import FeatureModel, SPECS, MEAN, STD, geometry, digest


def extract(model, dataset, output, batch_size, workers):
    """Write one float32 matrix and ordered image manifest per dataset class."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(model=model.name, weights_sha256=digest(model.weight_path),
                    preprocessing='resize256_bilinear_center224_imagenet_normalization',
                    classes=dataset.classes, class_to_idx=dataset.class_to_idx,
                    torch=torch.__version__, numpy=np.__version__, files={})
    (output / 'manifest.json').write_text(json.dumps({**manifest, 'complete': False}, indent=2), encoding='utf-8')
    grouped = [[] for _ in dataset.classes]
    for i, (_, label) in enumerate(dataset.samples):
        grouped[label].append(i)
    for class_id, synset in enumerate(dataset.classes):
        indices = grouped[class_id]
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                            num_workers=workers, shuffle=False)
        rows = []
        with torch.inference_mode():
            for images, _ in loader:
                rows.append(model.net(images.to(model.device)).cpu().numpy())
        matrix = np.concatenate(rows).astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError(f'Non-finite activations for {synset}')
        path = output / f'{synset}.pkl'
        with path.open('wb') as stream:
            pickle.dump(matrix, stream, protocol=4)
        manifest['files'][path.name] = dict(rows=len(matrix), sha256=digest(path),
            images=[Path(dataset.samples[i][0]).relative_to(dataset.root).as_posix() for i in indices])
        print(f'{class_id+1}/{len(dataset.classes)} {synset}: {len(matrix)} images', flush=True)
    manifest['complete'] = True
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SPECS, required=True)
    parser.add_argument('--images', type=Path, required=True, help='train/ or val/ with 1,000 synset directories')
    parser.add_argument('--weights', type=Path, default=Path('data/weights'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive and workers nonnegative')
    transform = transforms.Compose([geometry('resize256_bilinear'), transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = ImageFolder(args.images, transform=transform)
    if len(dataset.classes) != 1000:
        parser.error('Expected all 1,000 ImageNet synset directories in canonical sorted order')
    torch.set_num_threads(4)
    model = FeatureModel(args.model, args.weights, args.device)
    extract(model, dataset, args.output, args.batch_size, args.workers)


if __name__ == '__main__':
    main()
