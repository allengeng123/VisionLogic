"""Show every 8-connected component at 15% of each saved heatmap maximum."""
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from render_explanations import font

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/heatmap_components_15pct'
PALETTE = ['#16a34a', '#ed303b', '#2563eb', '#a855f7', '#e58a09', '#06a3b5']
SCALE = 3


def components(heat, fraction=.15):
    # Do not add a component-size filter or discard isolated surviving pixels.
    peak = float(heat.max())
    if not 0 < fraction <= 1:
        raise ValueError('Heatmap cutoff fraction must be in (0, 1]')
    mask = heat >= fraction * peak if peak > 0 else np.zeros(heat.shape, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = map(int, stats[k])
        boxes.append(dict(x0=x, y0=y, x1=x+w, y1=y+h, pixels=area))
    assert sum(b['pixels'] for b in boxes) == int(mask.sum())
    return mask, boxes, peak


def draw_boxes(image, layers):
    canvas = image.resize((image.width*SCALE, image.height*SCALE), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(canvas)
    for layer in layers:
        for b in layer['boxes']:
            xy = (b['x0']*SCALE, b['y0']*SCALE, b['x1']*SCALE-1, b['y1']*SCALE-1)
            d.rectangle(xy, outline=layer['color'], width=6)
    return canvas


def overlay(image, strength, color, opacity=.8):
    """Linear opacity shows H/max(H); preserve the original image underneath."""
    rgb = np.asarray(image, dtype=np.float32)
    alpha = opacity * np.asarray(strength, dtype=np.float32)[..., None]
    tint = np.array(ImageColor.getrgb(color), dtype=np.float32)
    result = np.clip(rgb * (1-alpha) + tint * alpha, 0, 255).astype('uint8')
    return Image.fromarray(result).resize((image.width*SCALE, image.height*SCALE), Image.Resampling.LANCZOS)


def main(source_path=None, output=None):
    global OUT
    if output is not None:
        OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(source_path or ROOT/'results/visual_explanations/manifest.json').read_text())['cases']
    groups = defaultdict(list)
    for r in source:
        digest = hashlib.sha256(Path(r['original_image']).read_bytes()).hexdigest()
        groups[(r['model'], r['class_id'], digest)].append(r)
    cards = []; records = []; previews = []; heat_previews = []
    names = {130: 'Flamingo', 277: 'Red fox', 360: 'Otter', 11: 'Goldfinch', 335: 'Fox squirrel', 497: 'Church'}
    preview_class = 335 if any(r['class_id'] == 335 for r in source) else 277
    for (model, cid, digest), rows in groups.items():
        dest = OUT/f'{model}_{cid}'; dest.mkdir(exist_ok=True)
        original = Image.open(rows[0]['original_image']).convert('RGB')
        original.save(dest/'original.png')
        layers = []; missing = []; singles = []
        for r in rows:
            p = r['predicate']; label = f"p_({p['physical_id']},{p['branch']})"
            maps_path = r.get('source_maps') or (str(Path(r['source_mask']).parent/'maps.npz') if r.get('source_mask') else None)
            if not maps_path:
                missing.append(label); continue
            with np.load(maps_path) as arrays:
                heat = arrays['heat']
            assert heat.shape == (original.height, original.width)
            mask, boxes, peak = components(heat)
            color = PALETTE[len(layers) % len(PALETTE)]
            layer = dict(predicate=p, label=label, color=color, boxes=boxes,
                         maximum=peak, cutoff=.15*peak, surviving_pixels=int(mask.sum()))
            layers.append(layer)
            boxed = draw_boxes(original, [layer])
            boxed.save(dest/f"predicate_{p['logical_id']}.png")
            normalized = heat/peak if peak > 0 else np.zeros_like(heat)
            heat_overlay = overlay(original, normalized, color)
            threshold_overlay = overlay(original, mask, color, opacity=.45)
            heat_overlay.save(dest/f"heatmap_{p['logical_id']}.png")
            threshold_overlay.save(dest/f"threshold_{p['logical_id']}.png")
            Image.fromarray(mask.astype('uint8')*255).save(dest/f"mask_{p['logical_id']}.png")
            panels = [('original.png','Original input'),(f'heatmap_{p["logical_id"]}.png','Heatmap over original'),(f'threshold_{p["logical_id"]}.png','Surviving pixels: H ≥ 0.15 max(H)'),(f'predicate_{p["logical_id"]}.png',f'All {len(boxes)} connected-region boxes')]
            figures = ''.join(f'<figure><img src="{dest.name}/{filename}"><figcaption>{caption}</figcaption></figure>' for filename,caption in panels)
            singles.append(f'<section class="predicate"><h3 style="color:{color}">{label}</h3><div class="sequence">{figures}</div><p class="scale"><span class="gradient" style="background:linear-gradient(to right,white,{color})"></span> Heatmap strength: 0 → max(H). Tint opacity increases linearly with H/max(H); each map uses its own maximum.</p></section>')
            if cid == preview_class and len(layers) == 1:
                tile = Image.new('RGB',(1440,430),'white'); d = ImageDraw.Draw(tile)
                d.text((10,5),f'{model.upper()} / {label}',font=font(22,True),fill=color)
                for idx,(im,caption) in enumerate(zip([original,heat_overlay,threshold_overlay,boxed],['Original','Heatmap overlay','Pixels above 15% of maximum','Connected-region boxes'])):
                    tile.paste(im.resize((350,350),Image.Resampling.LANCZOS),(idx*360,40))
                    d.text((idx*360+3,398),caption,font=font(17),fill='#172536')
                heat_previews.append(tile)
        composite = draw_boxes(original, layers); composite.save(dest/'boxes.png')
        legend = ''.join(f'<span style="color:{l["color"]}"><b>{l["label"]}</b>: {len(l["boxes"])} boxes</span> ' for l in layers)
        note = f'<p>Attribution unavailable (numerical convergence check failed): {", ".join(missing)}</p>' if missing else ''
        predicted = rows[0].get('predicted_class',rows[0]['predicate']['class_id'])
        predicted_name = names.get(predicted, 'Monastery' if predicted == 498 else str(predicted))
        context = f'<p>Image label: {names[cid]} (class {cid}). Model prediction and predicate vocabulary: {predicted_name} (class {predicted}).</p>'
        cards.append(f'<article data-model="{model}"><h2>{model.upper()} / {names[cid]}</h2>{context}{"".join(singles)}{note}<details><summary>Combined boxes for all displayed predicates</summary><div class="pair"><figure><img src="{dest.name}/original.png"><figcaption>Original input</figcaption></figure><figure><img src="{dest.name}/boxes.png"><figcaption>All predicates together</figcaption></figure></div><div class="legend">{legend}</div></details></article>')
        records.append(dict(model=model, class_id=cid, image_sha256=digest, layers=layers, missing=missing))
        if cid == preview_class:
            tile = Image.new('RGB',(672,760),'white'); tile.paste(composite,(0,40)); d=ImageDraw.Draw(tile)
            d.text((10,5),f'{model.upper()} / {names[cid]}',font=font(24,True),fill='#172536')
            for i,l in enumerate(layers):
                d.text((10+i*330,723),f'{l["label"]}: {len(l["boxes"])} boxes',font=font(19,True),fill=l['color'])
            previews.append(tile)
    preview = Image.new('RGB',(1344,1520),'white')
    for i,tile in enumerate(previews): preview.paste(tile,((i%2)*672,(i//2)*760))
    preview.save(OUT/'four_models_preview.png')
    heat_preview = Image.new('RGB',(1440,max(430,430*len(heat_previews))),'white')
    for i,tile in enumerate(heat_previews): heat_preview.paste(tile,(0,i*430))
    heat_preview.save(OUT/'heatmaps_to_boxes_preview.png')
    header = '''<!doctype html><meta charset="utf-8"><title>Heatmap connected-region boxes</title><style>
body{font:17px system-ui;background:#f3f5f7;color:#172536;margin:0}main{max-width:1150px;margin:30px auto;padding:0 20px}p{line-height:1.6}article{background:white;border-radius:12px;padding:22px;margin:22px 0}.pair{display:flex;flex-wrap:wrap;gap:20px}figure{margin:0;flex:1;min-width:250px}img{width:100%;display:block}figcaption{padding:10px 0;font-weight:600}.legend{display:flex;gap:25px;margin:14px 0}summary{cursor:pointer;padding:12px 0}select{font:inherit;padding:8px}.controls{position:sticky;top:0;background:#f3f5f7;padding:12px 0;z-index:1}[hidden]{display:none!important}</style><main>
<h1>Heatmap connected-region boxes</h1><p>Keep pixels at or above <b>15% of each predicate heatmap's maximum</b>, find 8-connected regions, and draw a tight box around every region. All components are included, even tiny ones. Green identifies the first displayed predicate; additional predicates use red, blue, and other colors. Each predicate keeps its color across its boxes and individual view.</p><p>These are initial heatmap proposals. This view does not perform ablation or iterative shrinking. It uses the saved neuron attribution heatmaps at the original 224-pixel model input resolution. Predicate labels omit the class shown in each example; only the available demo predicates are displayed.</p><div class="controls">Model <select id="model"><option value="all">All four models</option><option value="vit">ViT</option><option value="resnet">ResNet</option><option value="convnext">ConvNeXt</option><option value="swin">Swin</option></select></div>'''
    header += '<style>main{max-width:1600px}.sequence{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.sequence figure{min-width:0}.predicate{border-top:1px solid #e2e7ed;padding-top:10px;margin-top:22px}.scale{font-size:14px;color:#566474}.gradient{display:inline-block;width:95px;height:14px;vertical-align:middle;border:1px solid #ddd;margin-right:8px}@media(max-width:850px){.sequence{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>'
    footer = '''<script>document.querySelector('#model').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.model!==e.target.value);</script></main>'''
    (OUT/'index.html').write_text(header+''.join(cards)+footer,encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(dict(threshold_fraction=.15,connectivity=8,component_filter=None,cases=records),indent=2),encoding='utf-8')
    print(json.dumps(dict(images=len(records),predicates=sum(len(r['layers']) for r in records),gallery=str(OUT/'index.html'))))


if __name__ == '__main__': main()
