# Reproduction guide

Run commands from the repository root after following the README installation.
The single-image demo works on CPU. The historical random-100 benchmark requires
CUDA, ImageNet validation images, and the specified frozen checkpoints.

## Protocols

The release preserves two explicitly named protocols:

| Setting | `paper` | `legacy` |
| --- | --- | --- |
| Source images | Correctly classified training images in each dataset class | Majority-predicted rows within each class file; last group wins if predicted classes collide |
| Feature sorting | `np.argsort(contributions, kind='stable')[::-1]` for all models | Stable for ViT/ResNet; NumPy quicksort for ConvNeXt/Swin |
| Threshold fit | Minimum selected positive activation / maximum selected negative activation | Same extrema rule, on the legacy source population |
| Available fitted artifacts | Must be regenerated from training activations | Included in `results/section4_1/by_model/` |

`paper` follows the supplied September 2026 manuscript's source-selection and
sorting specification. It is the default for **new pathway and analysis runs**.
The bundled tables and grounding summary are frozen **legacy** experimental
artifacts; their reported numbers have not been re-established with `paper`.
Changing the source population or feature sorting can change thresholds, pathway
sizes, and grounding outcomes. Do not mix checkpoints/thresholds across protocols.

The legacy checkpoint reader treats old checkpoints without a `protocol` field
as `legacy`. New checkpoints and analysis summaries record the protocol. Analysis
rejects checkpoint/protocol mismatches. `explain.py` requires the generated
`summary.json` alongside paper thresholds and checks the protocol, model, and
checkpoint hash; keep that summary with newly fitted thresholds.

## 1. Prepare images and checkpoints

Obtain ImageNet-1k through its official access process and organize **both** train
and validation splits into the 1,000 synset directories described in [DATA.md](DATA.md).
Download the four exact checkpoints:

```bash
python download_weights.py
```

No training of the classifiers is performed. The checkpoint downloader verifies
the SHA-256 prefix published in the torchvision filename.

## 2. Extract final-head input activations

Example for ResNet-50; replace `resnet` with each model key to process all models:

```bash
python extract_activations.py --model resnet --images /path/to/imagenet/train --device cuda --output data/activations/resnet/train
python extract_activations.py --model resnet --images /path/to/imagenet/val --device cuda --output data/activations/resnet/val
```

The extractor uses the manuscript's grounding preprocessing: resize the shorter
side to 256 with bilinear interpolation, center-crop 224, and normalize with
ImageNet mean/std. It writes one float32 pickle per class and a manifest containing
ordered relative image paths, checkpoint/file hashes, and class IDs. The full
extractor refuses partial class vocabularies because sorted class order determines
the 1,000-class label indices. It refuses to overwrite an existing output folder.

These are newly generated features, not a bitwise recreation of the historical
saved activation matrices. The latter are not distributed; their exact extraction
provenance is not established by this release. Numeric results can also depend on
device, NumPy version, and floating-point accumulation.

Check the frozen-head reconstruction accuracy:

```bash
python check_saved_activation_accuracy.py --model resnet --data-dir data/activations/resnet/val --weights data/weights/resnet50-11ad3fa6.pth --output outputs/resnet-accuracy.json
```

The optional `--notebook` argument compares historical notebook transcripts;
it is not needed for accuracy computation or normal use.

## 3. Extract source pathways

```bash
python build_initial_pathway_checkpoints.py --models resnet --protocol paper --activation-root data/activations --weights-dir data/weights --output-dir outputs/paper-pathways
```

Omit `--models` to process all four models. The selector adds feature contributions
to all class logits and stops at the first prefix recovering the frozen prediction.
It does not apply an ordered-signature extension. Use a separate output directory
for `--protocol legacy`; the optional historical extended-checkpoint audit is
intended only for matching legacy inputs.

## 4. Fit predicates and analyze held-out data

```bash
python visionlogic_section42_predicate_analysis.py model --model resnet --protocol paper --train-dir data/activations/resnet/train --val-dir data/activations/resnet/val --weights data/weights/resnet50-11ad3fa6.pth --phase3 outputs/paper-pathways/phase3_initial_resnet.pkl --output-dir outputs/paper/resnet
```

Repeat for `vit`, `convnext`, and `swin`, using their checkpoint filenames. Then:

```bash
python visionlogic_section42_predicate_analysis.py combine --input-dir outputs/paper --output-dir outputs/paper/combined
```

For a single-model run, add `--models resnet` to `combine`. Outputs include
`predicate_metrics.csv.gz`, class/reuse tables, population summaries, and plots.
The script retains its historical `section42` filename; the current manuscript
places the feature-use analysis in Section 4.1.

Use fitted `paper` thresholds in the demo:

```bash
python explain.py --model resnet --protocol paper --predicates outputs/paper/resnet/predicate_metrics.csv.gz --image /path/to/image.jpg --device cuda --output outputs/paper-explanation
```

## 5. Historical random-100 grounding evaluation

This workflow uses the bundled **legacy** predicates and the canonical
`random.Random(42).sample(range(1, 50001), 100)` ImageNet validation IDs.
It retains all eligible predicates and failures. These scripts require CUDA.
Use fresh output directories; resume checks reject changes to protected inputs.

Set absolute paths before running (Bash):

```bash
export VISIONLOGIC_IMAGENET_VAL=/path/to/imagenet/val
export VISIONLOGIC_WEIGHTS="$PWD/data/weights"
export VISIONLOGIC_PREDICATES="$PWD/results/section4_1/by_model"
```

PowerShell equivalents:

```powershell
$env:VISIONLOGIC_IMAGENET_VAL = 'D:/imagenet/val'
$env:VISIONLOGIC_WEIGHTS = "$PWD/data/weights"
$env:VISIONLOGIC_PREDICATES = "$PWD/results/section4_1/by_model"
```

For the two architectures in the paper's grounding table:

```bash
python experimental_grounding/evaluate_grounding_random100.py --models vit resnet
python experimental_grounding/evaluate_gradcam_random100_four_models.py --model vit
python experimental_grounding/evaluate_gradcam_random100_four_models.py --model resnet
python experimental_grounding/extend_scorecam_lower010.py --model vit
python experimental_grounding/extend_scorecam_lower010.py --model resnet
python benchmark_report.py --models vit resnet --output outputs/grounding-summary.json
```

The first stage computes shared Score-CAM maps and saves the fixed image/predicate
population. The next scripts evaluate Grad-CAM and extend Score-CAM through 0.10.
The reporter combines recorded outcomes in **Grad-CAM-first** order, using
Score-CAM only after failure. The random control gets the corresponding two-stage
opportunity. Precomputing Score-CAM in this archival workflow does not represent
the runtime cost of the lazy fallback used by `explain.py`.

The reporter checks that both stages contain identical image/predicate keys and
thresholds. It labels partial runs and computes statistics from actual records,
including failure denominators and conditional accepted areas. To evaluate
ConvNeXt/Swin, run every stage for those models and pass them to the reporter.

The archived grounding configuration records Python 3.9.18, PyTorch 2.2.0+cu121,
torchvision 0.17.0+cu121, NumPy 1.24.3, Pillow 10.2.0, grad-cam 1.5.4, and
opencv-python 4.10.0.84 on an RTX 3070 Ti Laptop GPU. The current pinned
installation is newer and has been tested separately. Matching those original
versions is useful for numerical comparisons, but does not resolve the source
population/protocol difference above.

The older `report_gradcam_random100_four_models.py` contains historical narrative
and requires a separate historical audit summary. Use `benchmark_report.py` for
new runs; it does not repeat fixed historical conclusions.

## Validation limits

CPU regression tests do not require ImageNet. Real-image smoke checks establish
that loading a frozen checkpoint, selecting predicates, attribution, and blur
validation execute together. They do not reproduce the complete ImageNet analysis
or the 465-participant human study. Participant-level human-study data and its
analysis code are not included in this repository.
