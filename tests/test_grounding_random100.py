import unittest
import numpy as np
from PIL import Image
from core import Predicate
from evaluate_grounding_random100 import matched_dimensions, proposal_schedule, union_mask, run_search, CUTOFFS


class ConstantModel:
    def __init__(self, value):
        self.value = value
        self.calls = 0
    def evaluate(self, image):
        self.calls += 1
        return np.array([self.value]), 0


class GroundingEvaluationTests(unittest.TestCase):
    def test_area_bounds_and_rounding(self):
        for area in [1, 2, 15, 100, 224, 1000, 10001, 40000, 50175, 50176]:
            for ratio in [1/224, .1, .5, 1, 2, 10, 224]:
                w, h = matched_dimensions(area, ratio)
                self.assertTrue(1 <= w <= 224 and 1 <= h <= 224)
                self.assertLessEqual(abs(w*h-area), 112)
    def test_overlap_count(self):
        boxes = [dict(x0=0,y0=0,x1=10,y1=10),dict(x0=5,y0=0,x1=15,y1=10)]
        self.assertEqual(int(union_mask(boxes).sum()), 150)
    def test_disconnected_regions_and_seed(self):
        heat = np.zeros((224,224), np.float32)
        heat[5:15,5:15] = 1
        heat[50:65,70:80] = .8
        guided, random = proposal_schedule(heat, 42)
        self.assertEqual(len(guided), 10)
        self.assertEqual(len(guided[0]['boxes']), 2)
        self.assertEqual(guided[0]['pixels'], 250)
        self.assertEqual((guided, random), proposal_schedule(heat, 42))
        self.assertNotEqual(random, proposal_schedule(heat, 43)[1])
    def test_empty_kept_as_failure(self):
        guided, random = proposal_schedule(np.zeros((224,224),np.float32),42)
        for schedule in (guided, random):
            model = ConstantModel(-1)
            result = run_search(model,Predicate(0,0,0,'+',0),np.zeros((224,224,3),np.uint8),np.zeros((224,224,3),np.uint8),schedule)
            self.assertFalse(result['success'])
            self.assertEqual(len(result['attempts']),10)
            self.assertEqual(model.calls,0)
    def test_strict_deactivation_and_first_pass(self):
        guided,_ = proposal_schedule(np.ones((224,224),np.float32),42)
        for branch, value, passed in [('+',0,False),('+',-1,True),('-',0,False),('-',1,True)]:
            model = ConstantModel(value)
            original = np.zeros((224,224,3),np.uint8)
            result = run_search(model,Predicate(0,0,0,branch,0),original,original,guided)
            self.assertEqual(result['success'],passed)
            self.assertEqual(model.calls,1 if passed else 10)


if __name__ == '__main__':
    unittest.main()
