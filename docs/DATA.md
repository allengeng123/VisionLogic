# Data, models, and artifact formats

## ImageNet layout

ImageNet images are not redistributed. Obtain them from [ImageNet](https://www.image-net.org/)
and arrange the canonical 1,000 synsets in lexicographic order, matching the
torchvision checkpoint's class indices:

```text
imagenet/
  train/n01440764/*.JPEG
  train/n01443537/*.JPEG
  ...
  val/n01440764/ILSVRC2012_val_00000293.JPEG
  val/n01443537/*.JPEG
  ...
```

The validation archive originally has a flat layout. Organize it using the
official validation labels; do not guess labels from filenames. The random-100
evaluation requires the canonical `ILSVRC2012_val_XXXXXXXX.JPEG` filenames and
reports missing sampled IDs instead of resampling from available images.

## Frozen checkpoints

| Key | Architecture | Filename | Head input dimension |
| --- | --- | --- | ---: |
| `vit` | ViT-B/16 | `vit_b_16-c867db91.pth` | 768 |
| `resnet` | ResNet-50 | `resnet50-11ad3fa6.pth` | 2048 |
| `convnext` | ConvNeXt-Base | `convnext_base-6075fbad.pth` | 1024 |
| `swin` | Swin-T | `swin_t-704ceda3.pth` | 768 |

`download_weights.py` fetches these from `download.pytorch.org/models/` and checks
their SHA-256 filename prefixes. Do not substitute a checkpoint while reusing its
fitted thresholds. `FeatureModel` captures and verifies the exact input to the
final linear head before replacing that head with identity for attribution.

## Activations and pathways

```text
data/activations/<model>/train/<synset>.pkl
data/activations/<model>/val/<synset>.pkl
```

Each pickle is a two-dimensional float32 NumPy array `[images, features]`.
Only load pickle files you generated or trust. There must be one class file for
each of the 1,000 class indices; rows are in the order recorded by the extractor's
`manifest.json`. A manifest's `complete: true` is written only after all classes.

Pathway checkpoints store lists of signed feature IDs by source class. A positive
feature is `j`; a negative feature is `j + dimension`. The compatibility key
`resolved_pruned_dataset` stores initial prediction-preserving prefixes in this
release, despite its historical name. New checkpoints also contain `protocol`.

## Predicate tables

`predicate_metrics.csv.gz` contains source (`split=train`) and held-out summary
rows. The explanation command loads only source rows and identifies each predicate
by `(class_id, logical_id)`; fields include `physical_id`, `branch`, and `threshold`.
Positive predicates use `z[j] >= threshold`; negative predicates use
`z[j] <= threshold`. Equality remains active, so acceptance requires strict
deactivation after the blur intervention.

Bundled tables in `results/section4_1/by_model/` come from the **legacy** source
selection and sorting protocol. `results/grounding/paper_grounding_summary.json`
and `results/section4_1/model_summary_table.csv` preserve the earlier numerical
outputs. See [the protocol comparison](REPRODUCIBILITY.md#protocols) before using
them to make claims about the current manuscript specification.

## Storage and compute

Full train and validation extraction covers all images, without subsampling.
Activation storage is approximately `number_of_images * feature_dimension * 4`
bytes per model, plus manifests and pathway files. Grounding adds many network
forwards per predicate; Score-CAM processes all target-layer channels in batches
of eight. Use CUDA for the full benchmark and allow CPU demos more time.

## Optional display models

The paper uses foreground segmentation to visualize accepted regions. The tested
region remains the full box union. The current demo displays that union directly
and does not need BiRefNet, ISNet, LaMa, or their weights. Older scripts involving
those models are documented in [experimental_grounding/README.md](../experimental_grounding/README.md).

The MIT license applies to repository code. External datasets and checkpoints
remain governed by their respective licenses and access conditions.
