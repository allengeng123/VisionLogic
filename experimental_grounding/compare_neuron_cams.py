"""Compare library ScoreCAM and LayerCAM on fixed classifier-input neurons."""
from contextlib import redirect_stderr
import gc
import html
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_grad_cam import ScoreCAM, LayerCAM
from core import FeatureModel, Predicate, image_tensor, digest
from render_explanations import font

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/scorecam_layercam_neurons_v1'
SOURCE = ROOT/'results/heatmap_components_set2_15pct/source_manifest.json'
LAYERS = dict(vit='encoder.layers.encoder_layer_11.ln_1',resnet='layer4.2',convnext='features.7',swin='features.7.1.norm1')
NAMES = {11:'Goldfinch',335:'Fox squirrel',497:'Church',498:'Monastery'}


class SignedNeuronTarget:
    def __init__(self, predicate):
        self.index=predicate.physical_id;self.sign=predicate.sign
    def __call__(self, output):
        return self.sign * output[self.index]


def reshape_vit(tensor):
    assert tensor.ndim == 3 and tensor.shape[1] == 197
    return tensor[:,1:,:].reshape(tensor.shape[0],14,14,tensor.shape[2]).permute(0,3,1,2)


def reshape_swin(tensor):
    assert tensor.ndim == 4
    return tensor.permute(0,3,1,2)


def heat_images(image, cam):
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(cam,0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    overlay = np.clip(.55*np.asarray(image,dtype=np.float32)+.45*colored,0,255).astype('uint8')
    return Image.fromarray(colored),Image.fromarray(overlay)


def render(data):
    cards=[];previews=[]
    for r in data['cases']:
        p=r['predicate'];label=f"p_({p['physical_id']},{p['branch']})"
        folder=OUT/r['key']; cid=r['class_id']; pred=r['predicted_class']
        panels=[]
        for file,title in [('original.png','Original'),('scorecam_overlay.png','Score-CAM'),('layercam_overlay.png','LayerCAM')]:
            method = {'Score-CAM':'scorecam','LayerCAM':'layercam'}.get(title)
            if method and r[method]['zero_map']:
                title += ' (zero map at this layer)'
            panels.append(f'<figure><img src="{r["key"]}/{file}" alt="{title} {label}"><figcaption>{title}</figcaption></figure>')
        stats=' | '.join(f'{name}: {r[name]["seconds"]:.2f}s'+(' (zero map)' if r[name]['zero_map'] else '') for name in ('scorecam','layercam'))
        cards.append(f'<article data-model="{r["model"]}" data-class="{cid}"><h2>{r["model"].upper()} / {NAMES[cid]} / {label}</h2><p>Image class {cid}; model prediction: {NAMES.get(pred,pred)} ({pred}). Fixed neuron {p["physical_id"]}, branch {p["branch"]}; threshold {p["threshold"]:.6g}, original activation {r["activation"]:.6g}.</p><div class="row">{"".join(panels)}</div><p class="meta">Spatial layer: {html.escape(r["layer"])}; {r["native_shape"][-2]} × {r["native_shape"][-1]} grid. {stats}</p><details><summary>Heatmaps without image overlay</summary><div class="raw"><img src="{r["key"]}/scorecam_heat.png"><img src="{r["key"]}/layercam_heat.png"></div></details></article>')
        if cid == 335 and r['rank']==0:
            tile=Image.new('RGB',(1080,420),'white');d=ImageDraw.Draw(tile)
            d.text((8,5),f'{r["model"].upper()} / {label}',font=font(22,True),fill='#19324a')
            for i,(file,title) in enumerate([('original.png','Original'),('scorecam_overlay.png','Score-CAM'),('layercam_overlay.png','LayerCAM')]):
                im=Image.open(folder/file).resize((350,350),Image.Resampling.LANCZOS);tile.paste(im,(i*360,38))
                d.text((i*360+8,393),title,font=font(18,True),fill='#19324a')
            previews.append(tile)
    header='''<!doctype html><meta charset="utf-8"><title>Neuron heatmaps: Score-CAM vs LayerCAM</title>
<style>body{font:16px system-ui;background:#f2f5f8;color:#213245;margin:0}main{max-width:1400px;margin:28px auto;padding:0 20px}p{line-height:1.55}article{background:white;padding:22px;border-radius:12px;margin:24px 0}h2{font-size:23px}.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}figure{margin:0}img{width:100%;display:block}figcaption{padding:10px 0;font-weight:bold}.raw{display:flex;gap:20px}.raw img{width:45%}.meta{color:#637083;font-size:14px}.controls{position:sticky;top:0;background:#f2f5f8;padding:15px 0;z-index:1}select{font:inherit;padding:7px;margin-right:18px}.legend{display:inline-block;width:180px;height:14px;background:linear-gradient(to right,#30123b,#466be3,#28bbec,#32f298,#a4fc3c,#f9ba38,#ed4d0c,#7a0403);vertical-align:middle}summary{cursor:pointer;padding:8px 0}[hidden]{display:none!important}</style><main>
<h1>Neuron heatmaps: Score-CAM vs LayerCAM</h1><p>The same goldfinch, squirrel, and church images, using the exact same selected neuron predicates as demo set 2. Both methods target <b>+z[j]</b> or <b>−z[j]</b>, the signed input neuron of the frozen classifier head. The heatmap explains this continuous target; the class-specific threshold is recorded for context.</p>
<p>Original library implementations (pytorch-grad-cam 1.5.4), using the same spatial layer for both methods within each model. Score-CAM uses every channel and softmax weights over masked-input neuron scores. LayerCAM uses positive spatial gradients. No additional smoothing or thresholding is applied. These are heatmap proposals; no region validation is performed here.</p>
<p>Low <span class="legend"></span> High. Each heatmap is normalized separately to [0,1]; colors show relative spatial intensity, not comparable activation magnitudes. Overlays retain 55% of the original image. A zero map is reported explicitly.</p>
<div class="controls">Model <select id="model"><option value="all">All four models</option><option value="vit">ViT</option><option value="resnet">ResNet</option><option value="convnext">ConvNeXt</option><option value="swin">Swin</option></select>Image <select id="class"><option value="all">All three images</option><option value="11">Goldfinch</option><option value="335">Squirrel</option><option value="497">Church</option></select></div>'''
    footer='''<p><a href="results.json">Exact targets, settings and results</a> | <a href="../heatmap_components_set2_15pct/index.html">Earlier Integrated Gradients demo</a></p><script>function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=!['model','class'].every(k=>document.getElementById(k).value==='all'||a.dataset[k]===document.getElementById(k).value))}document.querySelectorAll('select').forEach(s=>s.onchange=filter)</script></main>'''
    (OUT/'index.html').write_text(header+''.join(cards)+footer,encoding='utf-8')
    if previews:
        sheet=Image.new('RGB',(1080,420*len(previews)),'white')
        for i,tile in enumerate(previews):sheet.paste(tile,(0,i*420))
        sheet.save(OUT/'four_models_preview.png')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads(SOURCE.read_text())
    data=dict(metadata=dict(source_sha256=digest(SOURCE),script_sha256=digest(__file__),grad_cam_version=importlib.metadata.version('grad-cam'),scorecam_source_sha256=digest(inspect.getfile(ScoreCAM)),layercam_source_sha256=digest(inspect.getfile(LayerCAM)),target='signed individual classifier-input neuron, not class logit or probability',target_layers=LAYERS,scorecam_batch_size=8,scorecam_channel_subsampling=False,aug_smooth=False,eigen_smooth=False,cam_normalization='library nonnegative clipping and min-max spatial normalization; resize to original 224 input',head_reconstruction_checks={}),cases=[])
    def save():
        (OUT/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8');render(data)
    save()
    for name in LAYERS:
        model=FeatureModel(name,Path(os.environ.get('VISIONLOGIC_WEIGHTS', ROOT.parent/'data/weights')),'cuda')
        data['metadata']['head_reconstruction_checks'][name]=model.head_reconstruction_max_error
        layer=model.net.get_submodule(LAYERS[name]);reshape={'vit':reshape_vit,'swin':reshape_swin}.get(name)
        for old in [r for r in source['cases'] if r['model']==name]:
            p=Predicate(**old['predicate']);target=SignedNeuronTarget(p)
            image=Image.open(old['original_image']).convert('RGB')
            z,predicted=model.evaluate(image)
            assert predicted==old['predicted_class'] and p.active(z[p.physical_id])
            np.testing.assert_allclose(z[p.physical_id],old['activation'],rtol=1e-4,atol=1e-5)
            key=f'{name}_{old["class_id"]}_{p.logical_id}';folder=OUT/key;folder.mkdir();image.save(folder/'original.png')
            record={k:old[k] for k in ('model','class_id','predicted_class','predicate','rank')}
            record.update(key=key,layer=LAYERS[name],activation=float(z[p.physical_id]),input_sha256=digest(old['original_image']))
            maps={}
            for title,cls in [('scorecam',ScoreCAM),('layercam',LayerCAM)]:
                x=image_tensor(image,'cuda').requires_grad_(title=='layercam')
                print('Running',name,old['class_id'],p.logical_id,title,flush=True);torch.cuda.synchronize();start=time.perf_counter()
                with cls(model=model.net,target_layers=[layer],reshape_transform=reshape) as cam:
                    cam.batch_size=8
                    with redirect_stderr(io.StringIO()):
                        heat=cam(input_tensor=x,targets=[target],aug_smooth=False,eigen_smooth=False)[0]
                    # The initial output is retained by BaseCAM even during Score-CAM masked forwards.
                    observed=float(target(cam.outputs[0]).detach().cpu())
                    np.testing.assert_allclose(observed,p.sign*z[p.physical_id],rtol=1e-4,atol=1e-5)
                    record['native_shape']=list(cam.activations_and_grads.activations[0].shape)
                    if title=='layercam':
                        grad=cam.activations_and_grads.gradients[0]
                        assert torch.isfinite(grad).all()
                        record['layer_gradient_abs_max']=float(grad.abs().max())
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                assert heat.shape==(224,224) and np.isfinite(heat).all()
                record[title]=dict(seconds=elapsed,minimum=float(heat.min()),maximum=float(heat.max()),zero_map=bool(heat.max()<=1e-8),signed_target=observed)
                maps[title]=heat
                raw,overlay=heat_images(image,heat);raw.save(folder/f'{title}_heat.png');overlay.save(folder/f'{title}_overlay.png')
                print('Finished',title,'seconds',round(elapsed,2),'range',float(heat.min()),float(heat.max()),flush=True)
                del x,cam;gc.collect();torch.cuda.empty_cache()
            np.savez_compressed(folder/'heatmaps.npz',**maps)
            data['cases'].append(record);save()
        del model;gc.collect();torch.cuda.empty_cache()
    print('Complete:',len(data['cases']),'neuron comparisons',OUT/'index.html',flush=True)


if __name__=='__main__':main()
