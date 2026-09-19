"""Four-model neuron Grad-CAM grounding evaluation at 0.15 and 0.10 bounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
from pytorch_grad_cam import GradCAM

from core import FeatureModel, Predicate, compose_edit, digest, image_tensor
from compare_neuron_cams import LAYERS, SignedNeuronTarget, reshape_swin, reshape_vit
from evaluate_grounding_random100 import OUT as ORIGINAL, WEIGHTS, matched_dimensions, union_mask
from render_heatmap_components import components

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/gradcam_random100_four_models_lower010_v1'
MODELS=('vit','resnet','convnext','swin')
CUTOFFS=[round(i/100,2) for i in range(60,9,-5)]


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def schedules(heat,seed):
    rng=np.random.default_rng(seed);guided=[];random=[]
    for cutoff in CUTOFFS:
        selected,boxes,_=components(heat,cutoff);area=int(union_mask(boxes).sum())
        assert not np.any(selected & ~union_mask(boxes))
        guided.append(dict(cutoff=cutoff,boxes=boxes,pixels=area))
        if area:
            bw=max(b['x1'] for b in boxes)-min(b['x0'] for b in boxes)
            bh=max(b['y1'] for b in boxes)-min(b['y0'] for b in boxes)
            w,h=matched_dimensions(area,bw/bh);x=int(rng.integers(0,224-w+1));y=int(rng.integers(0,224-h+1))
            boxes2=[dict(x0=x,y0=y,x1=x+w,y1=y+h)]
        else:boxes2=[];w=h=bw=bh=0
        random.append(dict(cutoff=cutoff,boxes=boxes2,pixels=w*h,matched_guided_pixels=area,
            area_error_pixels=w*h-area,guided_enclosing_aspect=bw/bh if bh else None))
    return guided,random


def run(model,p,rgb,blur,schedule):
    attempts=[]
    for proposal in schedule:
        mask=union_mask(proposal['boxes']);assert int(mask.sum())==proposal['pixels']
        attempt=dict(cutoff=proposal['cutoff'],pixels=proposal['pixels'],activation=None,flipped=False,evaluated=False)
        if mask.any():
            edited=compose_edit(rgb,blur,mask);assert np.array_equal(edited[~mask],rgb[~mask])
            values,pred=model.evaluate(Image.fromarray(edited));value=float(values[p.physical_id])
            attempt.update(activation=value,flipped=not p.active(value),evaluated=True,edited_prediction=pred,
                signed_margin=p.margin(value),edited_rgb_sha256=hashlib.sha256(edited.tobytes()).hexdigest())
        attempts.append(attempt)
        if attempt['flipped']:break
    success=attempts[-1]['flipped']
    return dict(success=success,accepted_step=len(attempts)-1 if success else None,
        accepted_area_pixels=attempts[-1]['pixels'] if success else None,attempts=attempts,
        model_evaluations=sum(a['evaluated'] for a in attempts))


def policy015(full):
    attempts=full['attempts'][:10]
    success=bool(attempts[-1]['flipped'])
    return dict(success=success,accepted_step=len(attempts)-1 if success else None,
        accepted_area_pixels=attempts[-1]['pixels'] if success else None,attempts=attempts,
        model_evaluations=sum(a['evaluated'] for a in attempts))


def summary(records):
    rows=[p for r in records for p in r['predicates']]
    result=dict(images=len(records),eligible_images=sum(bool(r['predicates']) for r in records),targets=len(rows),
        zero_maps=sum(p['zero_heatmap'] for p in rows))
    for policy in ('lower015','lower010'):
        result[policy]={}
        for method in ('guided','random'):
            wins=[p[policy][method] for p in rows if p[policy][method]['success']]
            areas=[p['accepted_area_pixels']/50176*100 for p in wins]
            result[policy][method]=dict(passed=len(wins),total=len(rows),rate=len(wins)/len(rows),
                median_area_percent=float(np.median(areas)) if areas else None,
                area_iqr_percent=[float(v) for v in np.percentile(areas,[25,75])] if areas else None,
                images_with_success=sum(any(p[policy][method]['success'] for p in r['predicates']) for r in records),
                model_evaluations=sum(p[policy][method]['model_evaluations'] for p in rows))
        result[policy]['paired_control']={k:sum((p[policy]['guided']['success'],p[policy]['random']['success'])==v for p in rows)
            for k,v in [('both',(True,True)),('guided_only',(True,False)),('random_only',(False,True)),('neither',(False,False))]}
    result['lower_bound_change']={k:sum((p['lower015']['guided']['success'],p['lower010']['guided']['success'])==v for p in rows)
        for k,v in [('already_passed',(True,True)),('new_at_010',(False,True)),('impossible_loss',(True,False)),('still_failed',(False,False))]}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True);args=parser.parse_args();name=args.model
    OUT.mkdir(parents=True,exist_ok=True);folder=OUT/name;folder.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    sample=json.loads((ORIGINAL/'sample.json').read_text())['images'];assert len(sample)==100
    sources=[Path(__file__),Path(__file__).with_name('core.py'),Path(__file__).with_name('compare_neuron_cams.py'),
        Path(__file__).with_name('evaluate_grounding_random100.py'),Path(__file__).with_name('render_heatmap_components.py')]
    protected=sources+[ORIGINAL/'summary.json']+[ORIGINAL/name/f'{e["image_id"]}.json' for e in sample]
    hashes={str(p):digest(p) for p in protected}
    config=dict(model=name,method='Library neuron-targeted Grad-CAM',layer=LAYERS[name],images=[e['image_id'] for e in sample],
        sample='Same canonical seed-42 random-100 images; no resampling',targets='All original J_c(x), exact signed classifier-head input neurons',
        cutoffs=CUTOFFS,reported_policies={'lower015':CUTOFFS[:10],'lower010':CUTOFFS},blur_radius=20,
        controls='Independently area-matched to Grad-CAM schedule; fixed original per-target seeds',
        fixed='Original predicted class, sign, physical neuron and no-extension source threshold',source_hashes=hashes)
    cfg=folder/'configuration.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config)
    model=FeatureModel(name,WEIGHTS,'cuda');reshape={'vit':reshape_vit,'swin':reshape_swin}.get(name);records=[]
    layer=model.net.get_submodule(LAYERS[name])
    for index,e in enumerate(sample):
        iid=e['image_id'];path=folder/f'{iid}.json'
        if path.exists():records.append(json.loads(path.read_text()));continue
        old=json.loads((ORIGINAL/name/f'{iid}.json').read_text());dest=folder/iid;dest.mkdir(exist_ok=True)
        image=Image.open(ORIGINAL/name/iid/'input.png').convert('RGB');image.save(dest/'input.png')
        z,pred=model.evaluate(image);assert pred==old['predicted_class'];np.testing.assert_allclose(z,np.load(ORIGINAL/name/iid/'original_activations.npz')['z'],rtol=1e-5,atol=1e-6)
        rgb=np.asarray(image);blur=np.asarray(image.filter(ImageFilter.GaussianBlur(20)));maps={};record=dict(image_id=iid,predicted_class=pred,predicates=[])
        started=time.perf_counter()
        for row in old['predicates']:
            p=Predicate(**row['predicate']);assert p.active(z[p.physical_id])
            with GradCAM(model=model.net,target_layers=[layer],reshape_transform=reshape) as cam:
                heat=cam(input_tensor=image_tensor(image,'cuda').requires_grad_(True),targets=[SignedNeuronTarget(p)])[0]
            assert np.isfinite(heat).all();maps[f'p{p.logical_id}']=heat
            pair=schedules(heat,int(row['control_seed']));out=dict(predicate=row['predicate'],original_activation=float(z[p.physical_id]),zero_heatmap=bool(heat.max()==0))
            out['lower010']={m:run(model,p,rgb,blur,s) for m,s in zip(('guided','random'),pair)}
            out['lower015']={m:policy015(out['lower010'][m]) for m in ('guided','random')}
            write(dest/f'p{p.logical_id}_geometry.json',dict(guided=pair[0],random=pair[1]));record['predicates'].append(out)
        record['seconds']=time.perf_counter()-started;np.savez_compressed(dest/'heatmaps.npz',**maps);write(path,record);records.append(record);write(folder/'summary.json',summary(records))
        print(f'{name} {index+1}/100 {iid} targets={len(record["predicates"])}',flush=True)
    assert hashes=={str(p):digest(p) for p in protected}
    write(folder/'completion.json',dict(complete=True,protected_inputs_unchanged=True,summary=summary(records)))
    print(json.dumps(summary(records),indent=2),flush=True)


if __name__=='__main__':main()
