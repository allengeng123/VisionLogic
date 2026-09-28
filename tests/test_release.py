import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experimental_grounding'))
from core import Predicate, original_prefix
from explain import ground
from visionlogic_section42_predicate_analysis import source_index, extract_path
from benchmark_report import summarize
from build_initial_pathway_checkpoints import build
from visionlogic_section42_predicate_analysis import analyze_model
from core import load_predicates
from extract_activations import extract


class ReleaseTests(unittest.TestCase):
    def test_extractor_preserves_class_and_image_order(self):
        class Features(torch.nn.Module):
            def forward(self, images):
                return images.mean((2,3))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'images'
            for name in ['b','a']:
                (images/name).mkdir(parents=True)
                Image.new('RGB', (8,8), (64,128,255)).save(images/name/'z.png')
                Image.new('RGB', (8,8), (0,0,0)).save(images/name/'a.png')
            weights = root/'weights'
            weights.write_bytes(b'test model')
            model = SimpleNamespace(name='test',weight_path=weights,net=Features(),device='cpu')
            output = root/'activations'
            extract(model, ImageFolder(images, transform=ToTensor()), output, 2, 0)
            with (output/'a.pkl').open('rb') as stream:
                values = pickle.load(stream)
            np.testing.assert_allclose(values[0], np.zeros(3))
            np.testing.assert_allclose(values[1], np.array([64,128,255])/255, atol=1e-6)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['classes'], ['a','b'])
            self.assertEqual(manifest['files']['a.pkl']['images'], ['a/a.png','a/z.png'])

    def test_pathway_to_threshold_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for i, values in enumerate([[[2,1,-1],[3,1,-1],[-4,1,-1]], [[-2,1,1],[-3,1,1],[5,1,1]]]):
                path = root / f'{i}.pkl'
                with path.open('wb') as stream:
                    pickle.dump(np.array(values, dtype=np.float32), stream)
                files.append(path)
            weights = root / 'weights.pth'
            torch.save({'fc.weight': torch.tensor([[1.,0.,0.],[-1.,0.,0.]]),
                        'fc.bias': torch.zeros(2)}, weights)
            checkpoint = root / 'pathways.pkl'
            with patch('build_initial_pathway_checkpoints.activation_files', return_value=files):
                build('resnet', root, weights, checkpoint, protocol='paper')
            args = SimpleNamespace(model='resnet',protocol='paper',weights=weights,
                train_dir=root,val_dir=root,phase3=checkpoint,output_dir=root/'analysis')
            with patch('visionlogic_section42_predicate_analysis.activation_files', return_value=files):
                self.assertEqual(analyze_model(args), 0)
                predicates = load_predicates(args.output_dir/'predicate_metrics.csv.gz')
                self.assertEqual(predicates[(0,0)].threshold, 2.)
                self.assertEqual(predicates[(1,3)].threshold, -2.)
                summary = json.loads((args.output_dir/'summary.json').read_text())
                self.assertEqual(summary['population']['phase3_source_rows'], 4)
                self.assertEqual(summary['protocol'], 'paper')
                args.protocol = 'legacy'
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    analyze_model(args)

    def test_source_protocols_keep_distinct_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i, values in enumerate([[[0,2],[1,0],[0,3]], [[0,2],[0,3]]]):
                path = Path(tmp) / f'{i}.pkl'
                with path.open('wb') as stream:
                    pickle.dump(np.array(values, dtype=np.float32), stream)
                files.append(path)
            paper, raw, kept = source_index(files, np.eye(2), np.zeros(2), 'paper')
            legacy, _, _ = source_index(files, np.eye(2), np.zeros(2), 'legacy')
            self.assertEqual((raw, kept), (5, 3))
            self.assertEqual(set(paper), {0,1})
            self.assertEqual(paper[0]['indices'].tolist(), [1])
            self.assertEqual(set(legacy), {1})
            self.assertEqual(legacy[1]['source_label'], 1)

    def test_stable_sort_for_all_paper_models(self):
        z = np.ones(40, dtype=np.float32)
        weights = np.stack([np.ones(40), np.zeros(40)])
        bias = np.array([0., 10.])
        expected = extract_path(z, 0, weights, bias, len(z), 'stable')
        for name in ('vit','resnet','convnext','swin'):
            self.assertEqual(original_prefix(z, 0, weights, bias, name, 'paper'), expected)

    def test_fallback_only_after_gradcam_failure(self):
        class Model:
            calls = 0
            def evaluate(self, image):
                self.calls += 1
                return np.array([0.]), 9
        rgb = np.ones((224,224,3), dtype=np.uint8)*255
        heat = np.zeros((224,224), dtype=np.float32)
        heat[10:20,10:20] = 1
        predicate = Predicate(1,0,0,'+',1.)
        fallback_calls = []
        def fallback():
            fallback_calls.append(True)
            return heat
        model = Model()
        stages, _, boxes = ground(model, predicate, rgb, np.zeros_like(rgb), heat, fallback)
        self.assertEqual(len(stages), 1)
        self.assertFalse(fallback_calls)
        self.assertTrue(boxes)
        stages, _, boxes = ground(model, predicate, rgb, np.zeros_like(rgb), np.zeros_like(heat), fallback)
        self.assertEqual([s['method'] for s in stages], ['gradcam','scorecam'])
        self.assertTrue(stages[-1]['success'])
        self.assertEqual(len(fallback_calls), 1)
        self.assertEqual(len(stages[0]['attempts']), 11)

    def test_failed_predicates_remain_in_output(self):
        rgb = np.zeros((224,224,3), dtype=np.uint8)
        heat = np.zeros((224,224), dtype=np.float32)
        stages, _, boxes = ground(None, Predicate(0,0,0,'+',1), rgb, rgb, heat, lambda: heat)
        self.assertEqual(len(stages), 2)
        self.assertFalse(stages[-1]['success'])
        self.assertFalse(boxes)

    def test_summary_mirrors_fallback_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            grad, score = Path(tmp)/'grad', Path(tmp)/'score'
            grad.mkdir(); score.mkdir()
            fail = dict(success=False, accepted_area_pixels=None, model_evaluations=11)
            win = dict(success=True, accepted_area_pixels=100, model_evaluations=1)
            predicate = dict(class_id=0,logical_id=0,physical_id=0,branch='+',threshold=1)
            row = dict(image_id='ILSVRC2012_val_00000001', predicates=[dict(predicate=predicate,
                original_activation=2,lower010=dict(guided=fail,random=win))])
            path = 'ILSVRC2012_val_00000001.json'
            (grad/path).write_text(json.dumps(row))
            row['predicates'][0]['lower010'] = dict(guided=win,random=fail)
            (score/path).write_text(json.dumps(row))
            result = summarize(grad, score)
            self.assertEqual(result['guided']['intervention_evaluations'], 12)
            self.assertEqual(result['random']['intervention_evaluations'], 1)
            self.assertEqual(result['guided']['successes'], 1)
            self.assertFalse(result['complete_100_images'])
            row['predicates'][0]['predicate']['threshold'] = 4
            (score/path).write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                summarize(grad, score)


if __name__ == '__main__':
    unittest.main()
