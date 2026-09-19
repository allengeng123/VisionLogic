"""Publication-style views of EXISTING tested masks; no new inference or edits.

Boxes frame up to three major connected groups. Every tested mask pixel remains
highlighted, including unboxed islands. A box is not a separately validated region.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
COLORS = {'vit': (132, 84, 184), 'resnet': (23, 138, 139),
          'convnext': (203, 125, 43), 'swin': (54, 118, 191)}
STYLE = dict(tint_opacity=0.16, faded_background_white=0.65,
             faded_cue_tint=0.08, box_margin_pixels=3,
             max_boxes=3, box_group_min_image_fraction=0.01,
             rendering_scale=3)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def font(size, bold=False):
    # Matplotlib ships portable font files; no dependency on user fonts.
    import matplotlib
    filename = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf'/filename),size)

def cue_boxes(mask):
    """Gentle geometric tightening around major groups; never mutate mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    order=sorted(range(1,n),key=lambda i:int(stats[i,cv2.CC_STAT_AREA]),reverse=True)
    boxes=[]
    h,w=mask.shape
    for i in order:
        x,y,bw,bh,area=map(int,stats[i])
        if boxes and area < STYLE['box_group_min_image_fraction']*mask.size:
            continue
        p=STYLE['box_margin_pixels']
        boxes.append(dict(x0=max(0,x-p), y0=max(0,y-p), x1=min(w,x+bw+p),
                          y1=min(h,y+bh+p), component_area=area))
        if len(boxes)>=STYLE['max_boxes']:
            break
    return boxes

def colored_view(image, mask, color, faded=False):
    original=np.asarray(image).astype(np.float32)
    if mask.shape != original.shape[:2]:
        raise ValueError('Mask/image coordinate mismatch')
    if faded:
        amount=STYLE['faded_background_white']
        canvas=(1-amount)*original+amount*255
        alpha=STYLE['faded_cue_tint']
    else:
        canvas=original.copy()
        alpha=STYLE['tint_opacity']
    canvas[mask]=(1-alpha)*original[mask]+alpha*np.asarray(color)
    return Image.fromarray(np.clip(np.rint(canvas),0,255).astype(np.uint8))

def draw_boxes(image, boxes, color):
    scale=STYLE['rendering_scale']
    out=image.resize((image.width*scale,image.height*scale),Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(out)
    for box in boxes:
        xy=(box['x0']*scale,box['y0']*scale,
            min(out.width-1,box['x1']*scale-1),min(out.height-1,box['y1']*scale-1))
        draw.rectangle(xy,outline=color,width=3)
    return out

def status_text(case):
    return {'single_fill_flip': 'Removal check passed (one LaMa fill)',
            'attribution_not_converged': 'Not tested: attribution did not converge'}.get(
                case['status'],'Removal check did not pass')

def make_panel(case, original, boxed, faded, path):
    size=448
    gap=24
    width=size*3+gap*4
    canvas=Image.new('RGB',(width,size+148),'white')
    draw=ImageDraw.Draw(canvas)
    color=COLORS[case['model']]
    p=case['predicate']
    op='≥' if p['branch']=='+' else '≤'
    draw.text((gap,15),f"{case['model'].upper()}  ·  neuron {p['physical_id']} {op} {p['threshold']:.3f}",
              font=font(23,True),fill=color)
    draw.text((gap,49),status_text(case),font=font(17),fill=(65,72,85))
    for i,(im,label) in enumerate(zip([original,boxed,faded],
                                     ['Original model input','Cue boxes + subtle overlay','Light background + cue regions'])):
        x=gap+i*(size+gap)
        draw.text((x,83),label,font=font(19),fill=(34,42,52))
        canvas.paste(im.resize((size,size),Image.Resampling.LANCZOS),(x,115))
    canvas.save(path)

def grid(cases, output):
    if not cases:
        return
    panels=[Image.open(p).convert('RGB') for p in cases]
    result=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)),'white')
    y=0
    for p in panels:
        result.paste(p,(0,y));y+=p.height
    result.save(output)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',type=Path,default=[ROOT/'results/demo_v2',ROOT/'results/demo_v2_rank1'])
    parser.add_argument('--output',type=Path,default=ROOT/'results/visual_explanations')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    immutable={}
    for run in args.runs:
        source=run/'results.json'
        immutable[str(source)]=sha(source)
        data=json.loads(source.read_text())
        for case in data['cases']:
            if 'predicate' not in case:
                continue
            folder=run/Path(case['figure']).parent
            image_path=folder/'original.png'
            original=Image.open(image_path).convert('RGB')
            attempts=[a for a in case['attempts'] if 'activation' in a]
            if attempts:
                last=attempts[-1]
                mask_path=folder/f"attempt_{last['attempt']}_mask.png"
                immutable[str(mask_path)]=sha(mask_path)
                mask=np.array(Image.open(mask_path))>0
            else:
                mask_path=None
                mask=np.zeros((original.height,original.width),dtype=bool)
            boxes=cue_boxes(mask)
            key=run.name+'_'+folder.name
            dest=args.output/key
            dest.mkdir(exist_ok=True)
            color=COLORS[case['model']]
            overlay=colored_view(original,mask,color)
            faded=colored_view(original,mask,color,True)
            boxed=draw_boxes(overlay,boxes,color)
            original.save(dest/'original.png')
            overlay.save(dest/'overlay.png')
            boxed.save(dest/'boxed.png')
            faded.resize(boxed.size,Image.Resampling.LANCZOS).save(dest/'faded.png')
            draw_boxes(faded,boxes,color).save(dest/'faded_boxes.png')
            make_panel(case,original,boxed,faded,dest/'comparison.png')
            record=dict(model=case['model'], class_id=case['class_id'], predicate=case['predicate'],
                        status=case['status'], label=status_text(case), run=run.name,
                        original_image=str(image_path.resolve()), source_mask=str(mask_path.resolve()) if mask_path else None,
                        source_mask_sha256=sha(mask_path) if mask_path else None, tested_mask_pixels=int(mask.sum()),
                        display_mask_pixels=int(mask.sum()), boxes=boxes,
                        boxed_component_pixels=sum(b['component_area'] for b in boxes),
                        key=key, diagnostic_url=f'../{run.name}/index.html',
                        interpretation='Complete tested mask is tinted; boxes guide attention to major groups, not separately validated regions.')
            records.append(record)
            (dest/'display.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    for path,expected in immutable.items():
        assert sha(path)==expected,'Rendering changed a measured artifact'
    showcase=[]
    for model in COLORS:
        options=[r for r in records if r['model']==model and r['class_id']==130]
        if options:
            showcase.append(args.output/options[-1]['key']/'comparison.png')
    grid(showcase,args.output/'four_models.png')
    passed=[r for r in records if r['status']=='single_fill_flip']
    grid([args.output/r['key']/'comparison.png' for r in passed],args.output/'passed_examples.png')
    (args.output/'manifest.json').write_text(json.dumps(dict(style=STYLE,cases=records,
        unchanged_source_artifacts=immutable),indent=2),encoding='utf-8')
    header='''<!doctype html><html><head><meta charset="utf-8"><title>VisionLogic · Visual explanations</title>
<style>
:root{color-scheme:light}body{font:16px system-ui;color:#263246;background:#f5f6f8;margin:0}main{max-width:1400px;margin:36px auto;padding:0 24px}
h1{font-size:32px;letter-spacing:-.7px;margin-bottom:10px}p{line-height:1.6}.intro{max-width:1000px;color:#596477}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;background:white;border:1px solid #e0e4e9;padding:16px;border-radius:12px;position:sticky;top:0;z-index:2}
select{font:inherit;padding:8px;border:1px solid #d5dbe2;border-radius:6px}label{display:flex;align-items:center;gap:8px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-top:24px}
article{background:white;border:1px solid #e0e4e9;border-radius:12px;overflow:hidden}article header{padding:18px 20px 10px}h2{font-size:20px;margin:0 0 8px}.badge{font-size:12px;padding:4px 8px;border-radius:4px;display:inline-block}.passed{background:#e4f3eb;color:#246645}.failed{background:#fff0d8;color:#896016}.none{background:#edf0f4;color:#657085}
article img{display:block;width:100%;aspect-ratio:1;object-fit:contain}article footer{padding:14px 20px;font-size:13px;color:#637082;line-height:1.6}a{color:#315f9f}.note{background:#fff8e9;border-left:3px solid #dab065;padding:12px 16px}small{color:#637082}button{cursor:pointer}summary{cursor:pointer;margin-top:24px;font-weight:600}.compare{width:100%;margin:15px 0} [hidden]{display:none!important}
</style></head><body><main>
<h1>VisionLogic · Visual explanations</h1>
<p class="intro">Compact cue boxes and translucent color on the original image—or a light background that leaves the cue details visible. These views use the existing tested masks; no new ablation or threshold fitting.</p>
<p class="note"><strong>5 passed removal checks, not 5 certified concepts.</strong> Passing means the fixed neuron predicate flipped after one LaMa replacement. Edit realism and semantic cue removal still need assessment. Boxes frame up to three major groups; every pixel of the tested mask remains tinted, including unboxed islands.</p>
<div class="controls"><label>Style <select id="style"><option value="boxed">Boxes + light overlay</option><option value="faded">Light background + cue regions</option><option value="faded_boxes">Light background + boxes</option><option value="overlay">Overlay only</option><option value="original">Original</option></select></label>
<label>Examples <select id="status"><option value="passed">Passed removal checks (5)</option><option value="all">All selected cases (21)</option><option value="failed">Unvalidated / not tested (16)</option></select></label>
<label>Model <select id="model"><option value="all">All four models</option><option value="vit">ViT</option><option value="resnet">ResNet</option><option value="convnext">ConvNeXt</option><option value="swin">Swin</option></select></label></div>
<p id="count"></p><div class="cards">'''
    cards=[]
    names={130:'Flamingo',277:'Red fox',360:'Otter'}
    for r in records:
        p=r['predicate'];op='≥' if p['branch']=='+' else '≤'
        status='passed' if r['status']=='single_fill_flip' else 'failed'
        cards.append(f'''<article data-model="{r['model']}" data-status="{status}" data-key="{r['key']}">
<header><h2>{r['model'].upper()} · {names.get(r['class_id'],r['class_id'])}</h2><span class="badge {status}">{html.escape(r['label'])}</span></header>
<img src="{r['key']}/boxed.png" alt="{r['model']} neuron {p['physical_id']} cue visualization">
<footer>Predicate: z[{p['physical_id']}] {op} {p['threshold']:.4f}<br>{len(r['boxes'])} display boxes · {r['tested_mask_pixels']} tested mask pixels<br>
<a href="{r['key']}/comparison.png">Compare both styles</a> · <a href="{r['diagnostic_url']}">Validation record</a></footer></article>''')
    tail='''</div><details><summary>Same flamingo image across all four models</summary><p>Second eligible predicate where available; first for ResNet. ResNet did not pass. This display selection does not depend on a successful outcome.</p><img class="compare" src="four_models.png"></details>
<p><a href="manifest.json">Mask provenance and box coordinates</a> · <a href="../analysis_v2/index.html">Original numerical demo</a></p>
<script>
const selectors=['style','status','model'].map(id=>document.getElementById(id));
function update(){const [style,status,model]=selectors.map(s=>s.value);let n=0;document.querySelectorAll('article').forEach(card=>{const visible=(status==='all'||card.dataset.status===status)&&(model==='all'||card.dataset.model===model);card.hidden=!visible;if(visible)n++;card.querySelector('img').src=card.dataset.key+'/'+style+'.png';});document.getElementById('count').textContent=n+' examples shown. Failed cases remain available under “All selected cases”.';}
selectors.forEach(s=>s.addEventListener('change',update));update();
</script></main></body></html>'''
    (args.output/'index.html').write_text(header+'\n'.join(cards)+tail,encoding='utf-8')
    print(f'Rendered {len(records)} cases ({len(passed)} passed). Source masks/results unchanged. Gallery: {args.output / "index.html"}')

if __name__=='__main__':
    main()
