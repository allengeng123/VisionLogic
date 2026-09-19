"""Segmentation-only comparison. Never runs classifiers or ablation trials."""
from __future__ import annotations

import html
import importlib.util
import json
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'assets/birefnet_dependencies'))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from torchvision import transforms

from core import ISNet, digest, geometry

OUT = ROOT / 'results/birefnet_isnet_random30'
POOL = Path(os.environ.get('VISIONLOGIC_IMAGENET_VAL', ROOT.parent/'data/imagenet-val'))
WEIGHTS = ROOT / 'assets/birefnet-general'
REVISION = 'e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4'
SEED = 20260915


def load_birefnet():
    # Execute the inspected, revision-pinned local architecture. No remote code fetch.
    package = types.ModuleType('birefnet_local')
    package.__path__ = [str(WEIGHTS)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location('birefnet_local.birefnet', WEIGHTS / 'birefnet.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.BiRefNet(config=module.BiRefNetConfig(bb_pretrained=False))
    status = model.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True)
    print('BiRefNet checkpoint: ' + str(status), flush=True)
    return model.eval().to('cuda')


def overlay(image, mask):
    rgb = np.asarray(image, dtype=np.float32)
    result = 0.4 * rgb + 0.6 * 255
    result[mask] = .50 * rgb[mask] + .45 * np.array([28, 190, 85]) + .05 * 255
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def save_visuals(folder, image, name, probability):
    assert probability.shape == (224, 224)
    assert np.isfinite(probability).all()
    assert probability.min() >= 0 and probability.max() <= 1
    np.save(folder / f'{name}_probability.npy', probability.astype(np.float32))
    mask = probability >= .5
    Image.fromarray(np.uint8(mask) * 255).save(folder / f'{name}_mask.png')
    Image.fromarray(np.uint8(np.rint(probability * 255))).save(folder / f'{name}_soft.png')
    overlay(image, mask).save(folder / f'{name}_overlay.png')
    return mask


def render(manifest, rows):
    cards = []
    for row in rows:
        key = row['key']
        paths = [('Input crop', 'input.png'), ('ISNet', 'isnet_overlay.png'), ('BiRefNet', 'birefnet_overlay.png')]
        panels = ''.join(f'<figure><figcaption>{label}</figcaption><a href="{key}/{path}"><img src="{key}/{path}" loading="lazy"></a></figure>' for label, path in paths)
        masks = ''.join(f'<figure><figcaption>{name}: binary mask</figcaption><img src="{key}/{name}_mask.png" loading="lazy"></figure>' for name in ['isnet', 'birefnet'])
        cards.append(f'<article><h2>{row["index"]:02d}. {html.escape(row["filename"])} <small>{row["synset"]}</small></h2><div class="grid">{panels}</div><p>Foreground pixels: ISNet {row["isnet_foreground"]:.1%}; BiRefNet {row["birefnet_foreground"]:.1%}. Mask overlap IoU: {row["pairwise_iou"]:.3f} (agreement, not accuracy).</p><details><summary>Inspect binary masks and uncropped source</summary><div class="grid">{masks}<figure><figcaption>Uncropped source</figcaption><img src="{key}/source.jpg" loading="lazy"></figure></div></details></article>')
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>BiRefNet vs ISNet: 30 random images</title><style>
body{font:16px system-ui,sans-serif;background:#f1f4f7;color:#172432;max-width:1150px;margin:32px auto;padding:0 20px}h1{font-size:30px}article,.intro{background:white;padding:22px;border-radius:12px;margin:24px 0}h2{font-size:19px}small{font-weight:400;color:#607080}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}figure{margin:0}figcaption{font-weight:600;margin:8px 0}img{width:100%;height:auto;image-rendering:auto}summary{cursor:pointer;color:#155b91}p{line-height:1.55}.note{border-left:4px solid #339b6c;padding-left:14px}@media(max-width:650px){.grid{gap:6px}body{padding:0 8px}article{padding:10px}}
</style></head><body><h1>BiRefNet vs ISNet</h1>'''
    page += f'''<section class="intro"><p>30 images sampled uniformly without replacement from {manifest['pool_size']:,} local ImageNet validation images, fixed seed {SEED}. No outcome-based filtering. Image index follows sampling order.</p><p>Identical inputs: resize shorter side to 256 (bilinear), center-crop 224. Each segmenter then uses its own 1024×1024 preprocessing. ISNet is the existing general-use ONNX adapter on CPU; BiRefNet is the official general-use safetensors model on CUDA, float32. Binary mask cutoff is 0.5 for both, without dilation or cleanup.</p><p>Green highlights predicted foreground; the rest is faded. These are <b>foreground masks, not predicate-specific explanations</b>. No segmentation reference labels are available. Pairwise IoU measures agreement only. Different inference backends mean timings are not a controlled speed comparison.</p><p class="note">Segmentation-only trial: no Score-CAM, box search, blur, or predicate tests run. Next grounding trial is recorded as blur radius 30, lower cutoff 0.15, <b>not executed</b>.</p><p><a href="manifest.json">Sampling and model provenance</a> · <a href="results.json">All mask statistics</a> · <a href="contact_sheet.jpg">30-image overview</a></p></section>'''
    (OUT / 'index.html').write_text(page + ''.join(cards) + '</body></html>', encoding='utf-8')
    # All 30 outcomes, in sampling order, arranged as six rows of five triplets.
    tile_w, tile_h = 384, 155
    sheet = Image.new('RGB', (5 * tile_w, 6 * tile_h), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    for i, row in enumerate(rows):
        x, y = (i % 5) * tile_w, (i // 5) * tile_h
        draw.text((x + 4, y + 3), f'{i+1:02d}    Original          ISNet             BiRefNet', fill='black', font=font)
        for j, f in enumerate(['input.png', 'isnet_overlay.png', 'birefnet_overlay.png']):
            im = Image.open(OUT / row['key'] / f).resize((124, 124), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + j * 128, y + 24))
    sheet.save(OUT / 'contact_sheet.jpg', quality=95)


def verify_results():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    rows = json.loads((OUT / 'results.json').read_text())
    pool = sorted(p for d in POOL.iterdir() if d.is_dir() for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    sample = random.Random(SEED).sample(pool, 30)
    assert [str(p) for p in sample] == [x['path'] for x in manifest['images']]
    assert len(rows) == len({r['key'] for r in rows}) == 30
    for row, source in zip(rows, manifest['images']):
        folder = OUT / row['key']
        original = Image.open(source['path']).convert('RGB')
        crop = np.asarray(geometry('resize256_bilinear')(original))
        assert np.array_equal(crop, np.asarray(Image.open(folder / 'input.png')))
        masks = []
        for name in ['isnet', 'birefnet']:
            probability = np.load(folder / f'{name}_probability.npy')
            mask = np.asarray(Image.open(folder / f'{name}_mask.png')) == 255
            assert probability.shape == mask.shape == (224, 224)
            assert np.isfinite(probability).all()
            assert np.array_equal(mask, probability >= .5)
            assert float(mask.mean()) == row[f'{name}_foreground']
            expected = np.asarray(overlay(Image.fromarray(crop), mask))
            assert np.array_equal(expected, np.asarray(Image.open(folder / f'{name}_overlay.png')))
            masks.append(mask)
        union = np.count_nonzero(masks[0] | masks[1])
        iou = float(np.count_nonzero(masks[0] & masks[1]) / union) if union else 1.
        assert iou == row['pairwise_iou']
    report = {'status': 'passed', 'images': 30,
              'checks': ['fixed-seed sample reproduction', 'exact input crop reproduction',
                         'all 60 probability arrays finite', 'saved binary masks match cutoff 0.5',
                         'all 60 overlays match saved masks', 'pairwise IoU and foreground statistics'],
              'pairwise_iou_ge_0_90': sum(r['pairwise_iou'] >= .9 for r in rows),
              'median_pairwise_iou': float(np.median([r['pairwise_iou'] for r in rows])),
              'segmentation_accuracy': 'not measured; no ground-truth masks',
              'blur30_trial': 'not executed'}
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['seed'] == SEED and len(manifest['images']) == 30
    else:
        pool = sorted(p for d in POOL.iterdir() if d.is_dir() for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        selected = random.Random(SEED).sample(pool, 30)
        manifest = {
            'seed': SEED, 'pool': str(POOL), 'pool_size': len(pool),
            'sampling': 'uniform random images without replacement; sorted path pool; no selection by model outcome',
            'input': 'RGB; shorter side 256 bilinear; center crop 224; no EXIF reorientation',
            'isnet': {'checkpoint': str(ROOT / 'assets/isnet-general-use.onnx'),
                      'sha256': digest(ROOT / 'assets/isnet-general-use.onnx'),
                      'adapter': 'unchanged core.ISNet; 1024 Lanczos; per-image max input normalization; output min-max; cv2 linear resize', 'backend': 'ONNX CPU'},
            'birefnet': {'repository': 'ZhengPeng7/BiRefNet', 'revision': REVISION,
                        'sha256': digest(WEIGHTS / 'model.safetensors'),
                        'architecture_sha256': digest(WEIGHTS / 'birefnet.py'),
                        'input': '1024 bilinear, ImageNet normalization',
                        'output': 'last decoder output sigmoid; float32 bilinear resize to 224; no min-max',
                        'backend': 'torch CUDA float32'},
            'binary_threshold': .5, 'dilation_pixels': 0,
            'next_grounding_trial': {'blur_radius': 30, 'lower_heatmap_cutoff': .15, 'status': 'planned_not_run'},
            'images': [{'path': str(p), 'sha256': digest(p)} for p in selected],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Sample fixed: {len(manifest["images"])} / {manifest["pool_size"]} images, seed {SEED}', flush=True)
    isnet = ISNet(ROOT / 'assets/isnet-general-use.onnx')
    birefnet = load_birefnet()
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    rows = []
    for index, source in enumerate(manifest['images'], 1):
        path = Path(source['path'])
        assert digest(path) == source['sha256']
        key = f'{index:02d}_{path.stem}'
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        metadata = folder / 'result.json'
        if metadata.exists():
            rows.append(json.loads(metadata.read_text()))
            print(f'{index}/30 cached {path.name}', flush=True)
            continue
        full = Image.open(path).convert('RGB')
        image = geometry('resize256_bilinear')(full)
        image.save(folder / 'input.png')
        full.save(folder / 'source.jpg', quality=95)
        start = time.perf_counter()
        ip = isnet(image)
        isnet_seconds = time.perf_counter() - start
        inputs = transform(image).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pred = birefnet(inputs)[-1].sigmoid()
            bp = torch.nn.functional.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        torch.cuda.synchronize()
        birefnet_seconds = time.perf_counter() - start
        im = save_visuals(folder, image, 'isnet', ip)
        bm = save_visuals(folder, image, 'birefnet', bp)
        union = np.count_nonzero(im | bm)
        row = {'index': index, 'key': key, 'filename': path.name, 'synset': path.parent.name,
               'input_sha256': digest(folder / 'input.png'),
               'isnet_foreground': float(im.mean()), 'birefnet_foreground': float(bm.mean()),
               'pairwise_iou': float(np.count_nonzero(im & bm) / union) if union else 1.,
               'isnet_seconds': isnet_seconds, 'birefnet_seconds': birefnet_seconds}
        metadata.write_text(json.dumps(row, indent=2), encoding='utf-8')
        rows.append(row)
        print(f'{index}/30 {path.name}: ISNet {im.mean():.1%}, BiRefNet {bm.mean():.1%}, IoU {row["pairwise_iou"]:.3f}', flush=True)
    (OUT / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    render(manifest, rows)
    verify_results()
    print(f'COMPLETE: {OUT / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
