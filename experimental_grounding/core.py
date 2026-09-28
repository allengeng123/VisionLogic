"""Experimental predicate grounding. Never refits thresholds or predicts with rules."""
from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode as I

SPECS = {
    'vit': ('vit_b_16', 'vit_b_16-c867db91.pth', 'heads.head'),
    'resnet': ('resnet50', 'resnet50-11ad3fa6.pth', 'fc'),
    'convnext': ('convnext_base', 'convnext_base-6075fbad.pth', 'classifier.2'),
    'swin': ('swin_t', 'swin_t-704ceda3.pth', 'head'),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def geometry(policy):
    options = {
        'resize256_bilinear': (256, I.BILINEAR),
        'resize232_bilinear': (232, I.BILINEAR),
        'resize232_bicubic': (232, I.BICUBIC),
        'resize256_bicubic': (256, I.BICUBIC),
        'square232_bilinear': ((232, 232), I.BILINEAR),
        'square232_bicubic': ((232, 232), I.BICUBIC),
        'square224_bilinear': ((224, 224), I.BILINEAR),
    }
    size, interpolation = options[policy]
    return transforms.Compose([transforms.Resize(size, interpolation=interpolation, antialias=True),
                               transforms.CenterCrop(224)])

def image_tensor(image, device='cpu'):
    return transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image)).unsqueeze(0).to(device)

class FeatureModel:
    """Replace ONLY the final Linear by Identity; forward returns its exact input."""
    def __init__(self, name, weight_dir, device='cpu'):
        constructor, filename, head_path = SPECS[name]
        self.name, self.device = name, torch.device(device)
        self.weight_path = Path(weight_dir) / filename
        self.net = getattr(models, constructor)(weights=None)
        self.net.load_state_dict(torch.load(self.weight_path, map_location='cpu', weights_only=True), strict=True)
        self.net.eval().to(self.device)
        self.head_path = head_path
        head = self.net.get_submodule(head_path)
        self.weight = head.weight.detach().clone()
        self.bias = head.bias.detach().clone()
        self.dimension = self.weight.shape[1]
        # Verify the captured feature reproduces the unmodified model output.
        captured = []
        handle = head.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224, device=self.device)
            expected = self.net(probe)
            rebuilt = torch.nn.functional.linear(captured[-1], self.weight, self.bias)
        handle.remove()
        torch.testing.assert_close(expected, rebuilt)
        parent_path, _, leaf = head_path.rpartition('.')
        parent = self.net.get_submodule(parent_path) if parent_path else self.net
        setattr(parent, leaf, torch.nn.Identity())
        with torch.no_grad():
            torch.testing.assert_close(self.net(probe), captured[-1])
        self.head_reconstruction_max_error = float((expected - rebuilt).abs().max())
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, image):
        z = self.net(image_tensor(image, self.device))[0]
        logits = torch.nn.functional.linear(z, self.weight, self.bias)
        return z.cpu().numpy(), int(logits.argmax())

@dataclass(frozen=True)
class Predicate:
    class_id: int
    logical_id: int
    physical_id: int
    branch: str
    threshold: float

    @property
    def sign(self):
        return 1 if self.branch == '+' else -1

    def active(self, value):
        return bool(value >= self.threshold if self.branch == '+' else value <= self.threshold)

    def margin(self, value):
        return self.sign * (float(value) - self.threshold)

def load_predicates(path):
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] != 'train':
                continue
            p = Predicate(int(row['class_id']), int(row['logical_id']), int(row['physical_id']),
                          row['branch'], float(row['threshold']))
            key = (p.class_id, p.logical_id)
            if key in result:
                raise ValueError('Duplicate source predicate')
            result[key] = p
    return result

def original_prefix(z, predicted, weights, bias, name, protocol='legacy'):
    """Same minimum-sufficient prefix/order as the existing Section 4.2 script."""
    if protocol not in ('paper', 'legacy'):
        raise ValueError('protocol must be paper or legacy')
    kind = 'stable' if protocol == 'paper' or name in ('vit', 'resnet') else 'quicksort'
    ranked = np.argsort(z * weights[predicted], kind=kind)[::-1]
    partial = bias.copy()
    result = []
    for j in ranked:
        result.append(int(j + (len(z) if z[j] < 0 else 0)))
        partial += weights[:, j] * z[j]
        if int(partial.argmax()) == predicted:
            break
    return result

def neuron_heatmap(model, image, predicate, steps=32, adaptive=True):
    """Captum IG of the head-input neuron, never of a class logit.

    Reference: each RGB channel's spatial mean, so evidence is relative to a
    spatially uniform version of this image. The IG path is not claimed to be ID.
    """
    from captum.attr import IntegratedGradients
    x = image_tensor(image, model.device)
    baseline = x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    with torch.no_grad():
        baseline_z = float(model.net(baseline)[0, predicate.physical_id])
        original_z = float(model.net(x)[0, predicate.physical_id])
    difference = original_z - baseline_z
    history = []
    schedule = [steps] if not adaptive else sorted(set([steps, max(steps,128), max(steps,512)]))
    for n_steps in schedule:
        attr, delta = IntegratedGradients(model.net).attribute(
            x, baselines=baseline, target=predicate.physical_id,
            n_steps=n_steps, internal_batch_size=2, return_convergence_delta=True)
        error = abs(float(delta[0]))
        converged = error <= max(1e-4, 0.05 * abs(difference))
        history.append(dict(steps=n_steps, completeness_error=error, converged=converged))
        if converged:
            break
    signed = predicate.sign * attr.detach()[0].sum(0).cpu().numpy()
    # Keep evidence supporting the predicate direction, not absolute gradients.
    positive = np.maximum(signed, 0)
    heat = cv2.GaussianBlur(positive, (0, 0), sigmaX=2)
    return heat, signed, dict(method='Captum IntegratedGradients on exact head-input neuron',
                              steps=n_steps, baseline='per-image spatial RGB mean',
                              baseline_activation=baseline_z, convergence_delta=float(delta[0]),
                              signed_attribution_sum=float(signed.sum()),
                              convergence_history=history, numerical_converged=converged,
                              completeness_tolerance='max(1e-4, 5% of abs(neuron difference))')

def mask_from_mass(heat, mass, foreground=None):
    """Retain all hot components; local ISNet boundary snap, not foreground veto."""
    if not 0 < mass <= 1:
        raise ValueError('mass must be in (0,1]')
    heat = np.asarray(heat, dtype=np.float32)
    if heat.ndim != 2 or not np.isfinite(heat).all() or np.any(heat < 0):
        raise ValueError('Expected finite nonnegative spatial heatmap')
    total = float(heat.sum())
    if total <= 0:
        return np.zeros(heat.shape, dtype=bool)
    ordered = np.sort(heat.ravel())[::-1]
    i = min(int(np.searchsorted(np.cumsum(ordered), mass * total)), len(ordered)-1)
    raw = (heat >= ordered[i]) & (heat > 0)
    # Close small raster gaps and enclosed holes before boundary assistance.
    # This is fixed local cleanup, not a per-component minimization search.
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE,
                          np.ones((5,5),np.uint8)).astype(bool)
    raw = binary_fill_holes(raw)
    # One fixed 3-pixel local band at 224 input resolution; never entire-object expansion.
    if foreground is not None:
        if foreground.shape != heat.shape:
            raise ValueError('ISNet mask and heatmap must share coordinates')
        kernel = np.ones((7, 7), np.uint8)
        fg = (foreground >= 0.5).astype(np.uint8)
        boundary = cv2.dilate(fg, kernel) != cv2.erode(fg, kernel)
        nearby = cv2.dilate(raw.astype(np.uint8), kernel).astype(bool)
        raw = (raw & ~boundary) | (nearby & boundary & fg.astype(bool))
    return raw

def compose_edit(original, replacement, mask):
    if original.shape != replacement.shape or mask.shape != original.shape[:2]:
        raise ValueError('Edit shape mismatch')
    result = original.copy()
    result[mask] = replacement[mask]
    assert np.array_equal(result[~mask], original[~mask])
    return result

class ISNet:
    """General-use ISNet ONNX export, preprocessing following rembg's adapter."""
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    def __call__(self, image):
        resized = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        array = np.array(resized).astype(np.float32)
        array /= max(float(array.max()), 1.0)
        array = (array - 0.5).transpose(2, 0, 1)[None]
        pred = self.session.run(None, {self.session.get_inputs()[0].name: array})[0][0, 0]
        pred = (pred - pred.min()) / max(float(pred.max()-pred.min()), 1e-8)
        return cv2.resize(pred, image.size, interpolation=cv2.INTER_LINEAR).clip(0, 1)

class Lama:
    """Published community TorchScript export. Deterministic: no fake seed repeats."""
    def __init__(self, path, device='cpu'):
        self.device = torch.device(device)
        self.net = torch.jit.load(str(path), map_location=self.device).eval()

    @torch.no_grad()
    def __call__(self, image, mask):
        original = np.asarray(image)
        x = torch.from_numpy(original.copy()).permute(2, 0, 1)[None].float().to(self.device)/255
        m = torch.from_numpy(mask.copy()).float()[None, None].to(self.device)
        h, w = mask.shape
        pad = (0, (-w) % 8, 0, (-h) % 8)
        result = self.net(torch.nn.functional.pad(x, pad, mode='reflect'),
                          torch.nn.functional.pad(m, pad, mode='reflect'))
        result = result[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
        result = np.clip(result*255, 0, 255).astype(np.uint8)
        return Image.fromarray(compose_edit(original, result, mask))
