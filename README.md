# VisionLogic

Code and compact result artifacts for **VisionLogic: Discovering and Grounding
Decision-Relevant Visual Concepts**.

VisionLogic is a post-hoc explanation framework for frozen ImageNet classifiers.
It extracts compact numerical pathways that reproduce each model prediction,
turns selected signed feature states into class-conditioned threshold predicates,
analyzes predicate use across images and classes, and grounds pathway predicates
with removal-based visual interventions.

## Final method in this repository

The repository implements the version used by the current paper:

1. For an input with penultimate representation `z`, decompose the predicted
   class logit into signed per-feature contributions.
2. Sort contributions with the model-specific ordering recorded in
   `section42_config.json` and retain the first prefix whose partial frozen-head
   logits reproduce the original prediction.
3. Fit class-conditioned predicates from selected source occurrences. Positive
   thresholds are source minima and negative thresholds are source maxima.
4. On a new image, use the original model prediction to select the class
   vocabulary and extract a fresh initial pathway. VisionLogic explains the
   prediction; it does not replace the classifier.
5. Ground active-and-selected predicates with neuron-targeted attribution.
   Grad-CAM proposes connected-region box unions first. If that search fails,
   Score-CAM supplies a second fixed proposal sequence. The first box union whose
   Gaussian-blur removal deactivates the unchanged predicate is accepted.
6. Foreground segmentation can be intersected with the accepted box for display.
   The intervention validates the full box, not the displayed intersection.

No ordered-signature extension, rule-classifier executor, retention/sufficiency
test, or bidirectional causal test is part of this release.

## Repository layout

| Path | Purpose |
| --- | --- |
| `build_initial_pathway_checkpoints.py` | Extract the prediction-preserving source pathways. Historical extended checkpoints are optional and used only for a prefix audit. |
| `visionlogic_section42_predicate_analysis.py` | Fit predicates, analyze source and held-out populations, and generate the Section 4.1 tables and figures. |
| `check_saved_activation_accuracy.py` | Reconstruct classifier logits from saved activations and the frozen final layer. |
| `experimental_grounding/` | Neuron attribution, proposal search, blur intervention, random controls, segmentation display, audits, and the grounding-flow figure. |
| `results/section4_1/` | Compact numerical outputs and predicate tables used by the paper. |
| `results/grounding/` | Paper-facing ViT and ResNet grounding summary. |
| `figures/` | Generated 1x4 analysis figures and the grounding-flow figure. |
| `tests/` | Unit tests for predicate logic, proposal geometry, controls, and stopping behavior. |

## Data not included

ImageNet images, saved activation matrices, torchvision weights, extracted
pathway checkpoints, and BiRefNet weights are not redistributed. Place them
outside Git and provide their locations through command-line arguments or the
environment variables below.

Expected activation layout:

```text
activation-root/
  vit/train/          vit/val/
  resnet/train/       resnet/val/
  convnext/train/     convnext/val/
  swin/train/         swin/val/
```

Each activation directory must contain 1,000 class files in the same sorted
class order used during extraction. Each file stores a two-dimensional NumPy
array of penultimate activations.

Expected torchvision weights:

```text
weights/
  vit_b_16-c867db91.pth
  resnet50-11ad3fa6.pth
  convnext_base-6075fbad.pth
  swin_t-704ceda3.pth
```

## Installation

Python 3.10 or 3.11 is recommended. Install a CUDA-enabled PyTorch build that
matches the cluster, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## 1. Verify saved activations

Run once per model:

```bash
python check_saved_activation_accuracy.py \
  --model vit \
  --data-dir /path/to/activation-root/vit/val \
  --weights /path/to/weights/vit_b_16-c867db91.pth \
  --notebook /path/to/visionlogic_vit.ipynb \
  --output outputs/activation_accuracy_vit.json
```

This recomputes top-1/top-5 predictions from `zW^T+b` and separately checks the
notebook's displayed per-class majority-consensus lines. Majority consensus is
not classifier accuracy.

## 2. Extract initial pathways

```bash
python build_initial_pathway_checkpoints.py \
  --activation-root /path/to/activation-root \
  --weights-dir /path/to/weights \
  --output-dir outputs/initial-pathways
```

The optional `--extended-checkpoint-dir` argument performs the historical-prefix
audit but does not alter extracted pathways.

## 3. Reproduce the numerical predicate analysis

Run each model separately. Example for ViT:

```bash
python visionlogic_section42_predicate_analysis.py model \
  --model vit \
  --train-dir /path/to/activation-root/vit/train \
  --val-dir /path/to/activation-root/vit/val \
  --weights /path/to/weights/vit_b_16-c867db91.pth \
  --phase3 outputs/initial-pathways/phase3_initial_vit.pkl \
  --output-dir outputs/section4_1/vit
```

Repeat for `resnet`, `convnext`, and `swin`, then combine:

```bash
python visionlogic_section42_predicate_analysis.py combine \
  --input-dir outputs/section4_1 \
  --output-dir outputs/section4_1/combined
```

The checked-in compact outputs report source/held-out pathway sizes, predicate
activation and selection frequencies, threshold retention, concentration, and
cross-class signed-feature reuse.

## 4. Reproduce grounding evaluation

Set the external inputs:

```bash
export VISIONLOGIC_IMAGENET_VAL=/path/to/imagenet-val
export VISIONLOGIC_WEIGHTS=/path/to/weights
export VISIONLOGIC_PREDICATES=$PWD/results/section4_1/by_model
```

Run scripts from `experimental_grounding` so their local imports resolve:

```bash
cd experimental_grounding

# Fixed Score-CAM proposal evaluation and matched random controls.
python evaluate_grounding_random100.py --models vit
python evaluate_grounding_random100.py --models resnet

# Neuron-targeted Grad-CAM proposal evaluation to the 0.10 lower bound.
python evaluate_gradcam_random100_four_models.py --model vit
python evaluate_gradcam_random100_four_models.py --model resnet

# Add Score-CAM proposals only for Grad-CAM failures and produce comparisons.
python extend_scorecam_lower010.py --model vit
python extend_scorecam_lower010.py --model resnet
python report_gradcam_random100_four_models.py
```

The acceptance rule always evaluates the same physical neuron, sign, and frozen
class-conditioned threshold. Attribution proposes regions; the removal test
determines acceptance. The matched random arm uses the same proposal areas and
two-stage opportunity.

The paper-facing summary is in
`results/grounding/paper_grounding_summary.json`. It records 274/276 successful
ViT predicate tests and 116/129 successful ResNet tests. At least one accepted
predicate occurred on 99/100 ViT images and 90/100 ResNet images. Median accepted
occupancy was 24.0% for ViT and 46.3% for ResNet, compared with 27.9% and 60.7%
for the matched random control.

Regenerate the Section 3.3 flow figure from the included verified example
panels and editable layout code:

```bash
python experimental_grounding/make_grounding_flow_figure.py
```

This writes both PDF and PNG versions under `figures/`.

## 5. Tests

```bash
export PYTHONPATH=$PWD/experimental_grounding
python -m unittest discover -s tests -p 'test_grounding_random100.py' -v
python -m unittest discover -s tests -p 'test_core.py' -v
```

Full numerical reproduction requires the external model and data artifacts.
Geometry and predicate tests run without ImageNet.

## Reproducibility notes

- Source and held-out populations are grouped by the frozen model prediction.
- Thresholds are fitted from selected source occurrences and frozen on held-out
  data.
- The source inequality is satisfied by construction because the threshold is an
  extremum over selected source occurrences.
- Identical signed features may support different class-conditioned predicates
  because thresholds are class specific.
- Cross-class reuse counts signed feature identities, not a single shared
  predicate.
- The grounding experiment is a one-way removal test. It does not claim that the
  accepted region is unique, pixel-minimal, sufficient in isolation, or a formal
  causal guarantee.
- Foreground segmentation is for visualization and does not replace the accepted
  box as the tested region.

## Paper result snapshot

The checked-in `model_summary_table.csv` reports the following medians:

| Model | Source active / selected | Held-out active / selected | Vocabulary eligibility | Conditional threshold retention |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 7 / 3 | 7 / 3 | 98.566% | 97.357% |
| ResNet-50 | 2 / 1 | 2 / 1 | 99.512% | 98.927% |
| ConvNeXt-Base | 9 / 4 | 9 / 4 | 98.920% | 98.359% |
| Swin-T | 15 / 7 | 15 / 7 | 99.468% | 98.428% |

The highlighted reuse findings are also retained: signed feature `(817,+)` in
ConvNeXt is selected in at least 1% of source examples in 534 classes, and signed
feature `(606,+)` in Swin in 327 classes.

## Citation

Citation metadata will be added when the paper becomes public. Until then,
please cite the repository and manuscript title in private research use.
