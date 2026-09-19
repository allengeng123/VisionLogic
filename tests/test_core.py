import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from core import Predicate, mask_from_mass, compose_edit, original_prefix, neuron_heatmap
from run_demo import run_removal, DEFAULTS

class Tests(unittest.TestCase):
    def test_signed_thresholds_and_equality(self):
        for branch in ('+', '-'):
            p = Predicate(0, 0, 0, branch, 2.)
            self.assertTrue(p.active(2.))
            self.assertFalse(p.active(2.-p.sign))
            self.assertEqual(p.margin(2.+p.sign), 1.)

    def test_disconnected_union_does_not_fill_gap(self):
        heat = np.zeros((40,40), np.float32)
        heat[5:10,5:10] = 1
        heat[30:35,30:35] = 1
        mask = mask_from_mass(heat, .6)
        self.assertTrue(mask[6,6] and mask[32,32])
        self.assertFalse(mask[20,20])

    def test_foreground_not_a_global_veto(self):
        heat = np.zeros((40,40), np.float32)
        heat[2:5,2:5] = 1
        fg = np.zeros_like(heat)
        fg[20:30,20:30] = 1
        self.assertTrue(mask_from_mass(heat,.6,fg)[3,3])
        self.assertFalse(mask_from_mass(heat,.6,fg)[25,25])

    def test_empty_and_bad_maps(self):
        self.assertFalse(mask_from_mass(np.zeros((5,5)),.6).any())
        with self.assertRaises(ValueError):
            mask_from_mass(np.full((5,5), np.nan),.6)

    def test_edit_preserves_outside(self):
        x=np.arange(75).reshape(5,5,3).astype(np.uint8)
        m=np.eye(5,dtype=bool)
        out=compose_edit(x,np.zeros_like(x),m)
        np.testing.assert_array_equal(out[~m],x[~m])
        self.assertEqual(out[m].sum(),0)

    def test_prefix_signed_ids(self):
        z=np.array([-2.,1.])
        w=np.array([[-2.,0.],[1.,0.]])
        self.assertEqual(original_prefix(z,0,w,np.zeros(2),'vit'),[2])

    def test_neuron_target_and_negative_sign(self):
        class Features(torch.nn.Module):
            def forward(self,x):
                # Distinct spatial evidence for two neurons, no class logits.
                return torch.stack((x[:,0,:2,:2].sum((1,2)),x[:,1,2:,2:].sum((1,2))),1)
        class Model:
            net=Features()
            device='cpu'
        array=np.zeros((4,4,3),dtype=np.uint8)
        array[:2,:2,0]=255
        image=Image.fromarray(array)
        p=Predicate(999,0,0,'+',0.)
        _,signed,info=neuron_heatmap(Model(),image,p,steps=8,adaptive=False)
        _,negative,_=neuron_heatmap(Model(),image,Predicate(1,2,0,'-',0.),steps=8,adaptive=False)
        np.testing.assert_allclose(negative,-signed)
        self.assertGreater(signed[:2,:2].sum(),0)
        self.assertEqual(float(signed[2:,2:].sum()),0)
        self.assertTrue(info['numerical_converged'])

    def test_matches_existing_prefix_implementation(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from visionlogic_section42_predicate_analysis import extract_path
        rng=np.random.RandomState(4)
        z=rng.randn(12).astype(np.float32)
        w=rng.randn(5,12).astype(np.float32)
        b=rng.randn(5).astype(np.float32)
        c=int((w@z+b).argmax())
        for name in ('vit','resnet','convnext','swin'):
            self.assertEqual(original_prefix(z,c,w,b,name),
                extract_path(z,c,w,b,len(z),'stable' if name in ('vit','resnet') else None))

    def test_stop_after_first_flip_and_negative_branch(self):
        class Model:
            def evaluate(self, image):
                return np.array([3.]), 999  # class change is irrelevant to acceptance
        calls=[]
        def replacement(image,mask):
            calls.append(mask)
            return Image.fromarray(compose_edit(np.array(image),np.zeros_like(image),mask))
        heat=np.zeros((32,32));heat[2:6,2:6]=1
        with tempfile.TemporaryDirectory() as folder:
            attempts,status,mask,_=run_removal(Model(),Image.new('RGB',(32,32),'white'),
                Predicate(0,1,0,'-',2.),heat,None,replacement,Path(folder),DEFAULTS)
        self.assertEqual(status,'single_fill_flip')
        self.assertEqual(len(calls),1)
        self.assertFalse(attempts[0]['predicate_active'])

    def test_area_guardrail_prevents_inpainting(self):
        def fail(*args):
            raise AssertionError('must not inpaint an overlarge mask')
        with tempfile.TemporaryDirectory() as folder:
            attempts,status,_,_=run_removal(None,Image.new('RGB',(32,32)),
                Predicate(0,0,0,'+',2.),np.ones((32,32)),None,fail,Path(folder),DEFAULTS)
        self.assertEqual(status,'unvalidated')
        self.assertEqual(attempts[0]['status'],'area_guardrail')

if __name__ == '__main__':
    unittest.main()
