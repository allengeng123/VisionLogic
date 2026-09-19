"""Pinned BiRefNet general-use inference, matching the random30 comparison."""
import torch
from torchvision import transforms
from compare_birefnet_isnet_random30 import load_birefnet, WEIGHTS, REVISION
from core import digest


class BiRefNetSegmenter:
    def __init__(self):
        self.model = load_birefnet()
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    @torch.inference_mode()
    def __call__(self, image):
        value = self.model(self.transform(image).unsqueeze(0).cuda())[-1].sigmoid()
        return torch.nn.functional.interpolate(
            value, size=(image.height, image.width), mode='bilinear',
            align_corners=False)[0, 0].cpu().numpy()

    @staticmethod
    def provenance():
        return dict(name='BiRefNet', checkpoint='ZhengPeng7/BiRefNet', revision=REVISION,
                    weights_sha256=digest(WEIGHTS/'model.safetensors'),
                    architecture_sha256=digest(WEIGHTS/'birefnet.py'),
                    threshold=.5, margin=0, precision='float32',
                    preprocessing='1024 bilinear resize, ImageNet normalization',
                    output='last output sigmoid, bilinear resize to input, no min-max',
                    role='visualization only; not an acceptance criterion')
