"""Score-CAM for every active source-vocabulary predicate; share masked forwards."""
from dataclasses import asdict
import colorsys
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.image import scale_cam_image
from core import FeatureModel, image_tensor, load_predicates, digest
from compare_neuron_cams import LAYERS, NAMES, reshape_vit, reshape_swin
from render_heatmap_components import components
from render_explanations import font

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/scorecam_all_active_15pct_v1'
PRIOR=ROOT/'results/scorecam_layercam_neurons_v1'
SOURCE=ROOT/'results/heatmap_components_set2_15pct/source_manifest.json'


def color(i):
    first=['#16a34a','#ef3038','#2563eb','#b43fe0','#ed820c','#059dac']
    if i<len(first):return first[i]
    hue=((i-len(first))*.61803398875+.12)%1
    rgb=colorsys.hls_to_rgb(hue,[.39,.49,.58][i%3],[.92,.72][i%2])
    return '#'+''.join(f'{round(x*255):02x}' for x in rgb)


@torch.no_grad()
def masked_scores(model,image):
    x=image_tensor(image,model.device);captured=[]
    reshape={'vit':reshape_vit,'swin':reshape_swin}.get(model.name,lambda t:t)
    layer=model.net.get_submodule(LAYERS[model.name])
    hook=layer.register_forward_hook(lambda mod,inputs,out:captured.append(reshape(out).detach().clone()))
    try:z=model.net(x)[0].cpu().numpy()
    finally:hook.remove()
    a=captured[0];native=a[0].cpu().numpy()
    # nn.UpsamplingBilinear2d in the library uses align_corners=True.
    up=torch.nn.functional.interpolate(a,size=x.shape[-2:],mode='bilinear',align_corners=True)
    mins=up.flatten(2).min(-1).values[:,:,None,None]
    maxs=up.flatten(2).max(-1).values[:,:,None,None]
    masks=(up-mins)/(maxs-mins+1e-8)
    outputs=[]
    for start in range(0,a.shape[1],8):
        batch=x*masks[0,start:start+8,None,:,:]
        outputs.append(model.net(batch).cpu().numpy())
    return z,native,np.concatenate(outputs,axis=0)


def map_for_predicate(native,scores,p):
    # Match the installed ScoreCAM channel softmax, weighted sum, clipping,
    # resize and second normalization. No channel sampling or new scoring.
    weights=torch.softmax(torch.from_numpy(p.sign*scores[:,p.physical_id]),dim=0).numpy()
    raw=(weights[:,None,None]*native).sum(axis=0)
    positive=np.maximum(raw,0)
    resized=scale_cam_image(positive[None],(224,224))
    result=scale_cam_image(np.maximum(resized,0))[0]
    return result,raw


def publish(data):
    (OUT/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    embedded=json.dumps(data['images']).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>Score-CAM: all active predicates</title>
<style>body{font:16px system-ui;color:#203246;background:#f2f5f8;margin:0}main{max-width:1450px;margin:28px auto;padding:0 20px}p{line-height:1.55}article{background:white;border-radius:12px;padding:22px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}canvas,img{width:100%;display:block}figure{margin:0}figcaption{font-weight:600;margin:10px 0}.filters{position:sticky;top:0;background:#f2f5f8;padding:12px 0;z-index:2}select,input,button{font:inherit;padding:8px;margin:6px 10px 6px 0}select{max-width:100%}.legend{display:flex;flex-wrap:wrap;gap:8px;max-height:190px;overflow:auto;padding:10px 0}.tag{font-size:13px;border:1px solid #ddd;border-left:7px solid;padding:5px;cursor:pointer;background:white}.details{color:#576574;font-size:14px}.status{padding:10px;background:#eef4f8}h2{margin-bottom:8px}[hidden]{display:none!important}@media(max-width:750px){.pair{grid-template-columns:1fr}}</style><main>
<h1>Score-CAM boxes for all active predicates</h1><p>For each image, we check <b>every source-calibrated predicate in the original model-predicted class vocabulary</b>. Both signed branches use their original inclusive thresholds. For each active predicate, retain Score-CAM pixels satisfying <b>H ≥ 0.15 max(H)</b>, find all 8-connected regions, and draw their tight boxes. Every nonempty component is included; outlines are thick and each predicate has its own color.</p>
<p>All active predicates are selected by default. Use the selector or click a colored predicate label to inspect its boxes alone. Zero Score-CAM maps stay listed, with no boxes. These are heatmap-derived proposals before any removal test or shrinking. With many active predicates, the combined view can be crowded.</p>
<p class="details">Same Score-CAM layers and signed neuron targets as the preceding comparison. All feature-map channels are used. Masked-image forwards are shared across neurons, then the original Score-CAM weights and map normalization are evaluated for each neuron independently. Predicates with different class thresholds remain distinct.</p>
<div class="filters">Model <select id="models"><option value="all">All models</option><option value="vit">ViT</option><option value="resnet">ResNet</option><option value="convnext">ConvNeXt</option><option value="swin">Swin</option></select>Image <select id="classes"><option value="all">All images</option><option value="11">Goldfinch</option><option value="335">Squirrel</option><option value="497">Church</option></select></div><div id="cards"></div><p><a href="results.json">All exact predicates and boxes</a> | <a href="../scorecam_layercam_neurons_v1/index.html">Score-CAM / LayerCAM comparison</a></p></main><script>
const data=DATA;
const modelSelect=document.getElementById('models'),classSelect=document.getElementById('classes');
modelSelect.replaceChildren(new Option('All models','all'));classSelect.replaceChildren(new Option('All images','all'));
for(const model of [...new Set(data.map(r=>r.model))])modelSelect.add(new Option(model==='vit'?'ViT':model==='resnet'?'ResNet':model.toUpperCase(),model));
for(const cid of [...new Set(data.map(r=>r.class_id))])classSelect.add(new Option(data.find(r=>r.class_id===cid).image_name,String(cid)));
const cards=document.getElementById('cards');
for(const r of data){
 const a=document.createElement('article');a.dataset.model=r.model;a.dataset.class=r.class_id;
 a.innerHTML=`<h2>${r.model.toUpperCase()} / ${r.image_name}</h2><p>Model prediction: ${r.predicted_name} (class ${r.predicted_class}). <b>${r.active_count} active / ${r.vocabulary_count} class predicates</b>; ${r.nonzero_maps} nonzero maps, ${r.zero_maps} zero maps; ${r.total_boxes} boxes.</p><div class="pair"><figure><img src="${r.key}/original.png"><figcaption>Original input</figcaption></figure><figure><canvas width="672" height="672"></canvas><figcaption>All active predicates: connected-region boxes</figcaption></figure></div><label>Show <select class="predicates"><option value="all">All ${r.active_count} active predicates</option></select></label><p class="status"></p><div class="legend"></div>`;
 cards.appendChild(a);const s=a.querySelector('.predicates'),legend=a.querySelector('.legend'),canvas=a.querySelector('canvas'),ctx=canvas.getContext('2d'),pic=new Image();
 for(let i=0;i<r.predicates.length;i++){const p=r.predicates[i],label=`p_(${p.physical_id},${p.branch})`;
 const option=new Option(`${label}: ${p.boxes.length} boxes${p.zero_map?' [zero map]':''}`,String(i));s.add(option);
 const tag=document.createElement('button');tag.className='tag';tag.style.borderLeftColor=p.color;tag.textContent=label;tag.title=`class ${p.class_id}; threshold ${p.threshold}; ${p.boxes.length} boxes`;tag.onclick=()=>{s.value=String(i);draw()};legend.appendChild(tag)}
 function draw(){ctx.clearRect(0,0,672,672);ctx.drawImage(pic,0,0,672,672);const selected=s.value==='all'?r.predicates:[r.predicates[Number(s.value)]];let count=0;
 for(const p of selected){ctx.strokeStyle=p.color;ctx.lineWidth=6;for(const b of p.boxes){const x=b.x0*3,y=b.y0*3,w=(b.x1-b.x0)*3,h=(b.y1-b.y0)*3;ctx.strokeRect(x+Math.min(3,w/2),y+Math.min(3,h/2),Math.max(0,w-6),Math.max(0,h-6));if(w<=6||h<=6){ctx.fillStyle=p.color;ctx.fillRect(x,y,w,h)}count++}}
 a.querySelectorAll('figcaption')[1].textContent=s.value==='all'?'All active predicates: connected-region boxes':`p_(${selected[0].physical_id},${selected[0].branch}): connected-region boxes`;
 a.querySelector('.status').textContent=s.value==='all'?`Showing all ${selected.length} active predicates, ${count} boxes.`:`Class ${selected[0].class_id}; z[${selected[0].physical_id}] ${selected[0].branch==='+'?'≥':'≤'} ${selected[0].threshold}; original activation ${selected[0].activation}; ${count} boxes${selected[0].zero_map?' (Score-CAM returned zero at this layer)':''}.`}
 s.onchange=draw;pic.onload=draw;pic.src=`${r.key}/original.png`;
 const evidence=document.createElement('details');evidence.innerHTML='<summary>Activation checks for this specific image</summary><p>Every listed predicate satisfies its original threshold on this image. Displayed values are rounded; the checks use full precision.</p><table style="width:100%;text-align:left"><thead><tr><th>Predicate</th><th>Activation</th><th>Condition</th><th>Active</th></tr></thead><tbody>'+r.predicates.map(p=>`<tr><td style="color:${p.color}">p_(${p.physical_id},${p.branch})</td><td>${p.activation.toPrecision(6)}</td><td>${p.branch==='+'?'≥':'≤'} ${p.threshold.toPrecision(6)}</td><td>${(p.branch==='+'?p.activation>=p.threshold:p.activation<=p.threshold)?'Yes':'No'}</td></tr>`).join('')+'</tbody></table>';a.appendChild(evidence);
}
function filter(){for(const a of cards.children)a.hidden=!(document.getElementById('models').value==='all'||a.dataset.model===document.getElementById('models').value)||!(document.getElementById('classes').value==='all'||a.dataset.class===document.getElementById('classes').value)}document.getElementById('models').onchange=filter;document.getElementById('classes').onchange=filter;
</script>'''
    (OUT/'index.html').write_text(page.replace('DATA',embedded),encoding='utf-8')


def refresh_display(data):
    for entry in data['images']:
        folder=OUT/entry['key'];image=Image.open(folder/'original.png').convert('RGB')
        overview=image.resize((672,672),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(overview)
        for i,r in enumerate(entry['predicates']):
            r['color']=color(i)
            for b in r['boxes']:draw.rectangle((b['x0']*3,b['y0']*3,b['x1']*3-1,b['y1']*3-1),outline=r['color'],width=6)
        overview.save(folder/'all_boxes.png')
    publish(data)


def main(output=None, source_path=None, models=None):
    global OUT,SOURCE
    if output is not None:OUT=Path(output)
    if source_path is not None:SOURCE=Path(source_path)
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads(SOURCE.read_text());prior=json.loads((PRIOR/'results.json').read_text())
    NAMES.update({int(k):v for k,v in source.get('image_names',{}).items()})
    data=dict(metadata=dict(source_sha256=digest(SOURCE),script_sha256=digest(__file__),layers=LAYERS,threshold_fraction=.15,connectivity=8,all_active=True,component_size_filter=None,selection='all predicates in predicted-class source vocabulary whose inclusive threshold holds',source_predicate_sha256={},library_comparison=[]),images=[])
    publish(data)
    for name in (models or LAYERS):
        model=FeatureModel(name,Path(os.environ.get('VISIONLOGIC_WEIGHTS', ROOT.parent/'data/weights')),'cuda')
        path=ROOT.parent/f'section42-results/results/{name}/predicate_metrics.csv.gz'
        vocab=load_predicates(path);data['metadata']['source_predicate_sha256'][name]=digest(path)
        for item in source['image_results']:
            if item['model']!=name:continue
            old=next(r for r in source['cases'] if r['model']==name and r['class_id']==item['class_id'])
            image=Image.open(old['original_image']).convert('RGB');z,pred=model.evaluate(image)
            population=sorted([p for (c,lid),p in vocab.items() if c==pred],key=lambda p:p.logical_id)
            active=[p for p in population if p.active(z[p.physical_id])]
            print(name,item['class_id'],'active',len(active),'/',len(population),'masked forwards...',flush=True)
            start=time.perf_counter();z_check,native,scores=masked_scores(model,image)
            np.testing.assert_allclose(z_check,z,rtol=1e-5,atol=1e-6)
            key=f'{name}_{item["class_id"]}';folder=OUT/key;folder.mkdir();image.save(folder/'original.png')
            np.savez_compressed(folder/'shared_scores.npz',native_activations=native,masked_neurons=scores,original_neurons=z)
            records=[];raw_maps=[]
            palette={p.logical_id:color(i) for i,p in enumerate(active)}
            for p in active:
                heat,raw=map_for_predicate(native,scores,p)
                mask,boxes,peak=components(heat)
                assert np.isfinite(heat).all()
                r=dict(**asdict(p),color=palette[p.logical_id],activation=float(z[p.physical_id]),zero_map=bool(peak==0),heat_maximum=peak,heat_cutoff=.15*peak,surviving_pixels=int(mask.sum()),boxes=boxes)
                records.append(r);raw_maps.append(raw)
                previous=next((v for v in prior['cases'] if v['model']==name and v['class_id']==item['class_id'] and v['predicate']['logical_id']==p.logical_id),None)
                if previous:
                    with np.load(PRIOR/previous['key']/'heatmaps.npz') as maps:reference=maps['scorecam']
                    error=float(np.max(np.abs(reference-heat)))
                    np.testing.assert_allclose(heat,reference,rtol=1e-4,atol=2e-5)
                    data['metadata']['library_comparison'].append(dict(key=previous['key'],max_abs_error=error))
            np.savez_compressed(folder/'native_maps.npz',raw=np.asarray(raw_maps),logical_ids=np.asarray([p.logical_id for p in active]))
            overview=image.resize((672,672),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(overview)
            for r in records:
                for b in r['boxes']:draw.rectangle((b['x0']*3,b['y0']*3,b['x1']*3-1,b['y1']*3-1),outline=r['color'],width=6)
            overview.save(folder/'all_boxes.png')
            entry=dict(key=key,model=name,class_id=item['class_id'],image_name=NAMES[item['class_id']],predicted_class=pred,predicted_name=NAMES.get(pred,str(pred)),vocabulary_count=len(population),active_count=len(active),nonzero_maps=sum(not r['zero_map'] for r in records),zero_maps=sum(r['zero_map'] for r in records),total_boxes=sum(len(r['boxes']) for r in records),native_shape=list(native.shape),seconds=time.perf_counter()-start,predicates=records)
            data['images'].append(entry);publish(data)
            print('Done',name,item['class_id'],'boxes',entry['total_boxes'],'zero maps',entry['zero_maps'],'seconds',round(entry['seconds'],1),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    print('Complete',len(data['images']),'images;',sum(r['active_count'] for r in data['images']),'active predicates;',len(data['metadata']['library_comparison']),'library matches',flush=True)


if __name__=='__main__':main()
