# Release validation

Validation was performed in a fresh Python 3.12 environment with the pinned
requirements and PyTorch 2.5.1+cpu / torchvision 0.20.1+cpu on Windows.

- `python -m pip check`: no broken requirements.
- `python run_tests.py`: 22 CPU tests, including a synthetic activation-to-pathway-to-threshold integration test and class/image-order checks for activation extraction.
- All ten documented CLI entry points accept `--help` without requiring private artifacts.
- The single-image command ran with the exact frozen checkpoints for all four architectures on `figures/grounding_flow_assets/input.png` using the bundled legacy thresholds.

| Model | Eligible predicates | Accepted | Predicates reaching Score-CAM |
| --- | ---: | ---: | ---: |
| ResNet-50 | 1 | 1 | 0 |
| ViT-B/16 | 3 | 3 | 0 |
| Swin-T | 10 | 9 | 4 |
| ConvNeXt-Base | 4 | 4 | 0 |

These are execution smoke checks on one image, **not benchmark results**.
The Swin run exercised real CPU Score-CAM fallback and retained a failed predicate.
The new benchmark reporter was also run against the original saved random-100
records: it reproduced the archived ViT count of 274/276 and ResNet count of
116/129, along with their matched-random counts and accepted-area medians.
This checks aggregation, not a fresh replay of the benchmark's network forwards.

The full ImageNet analysis under the new `paper` protocol and the human evaluation
have not been rerun as part of this release. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the manuscript/legacy distinction. GitHub Actions separately checks the CPU
suite on Ubuntu with Python 3.11 on each push and pull request.
