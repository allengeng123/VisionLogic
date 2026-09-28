# Grounding implementation and historical experiments

Use **`python explain.py` from the repository root** for a new image. It implements
the current Grad-CAM-first / Score-CAM-fallback / radius-20 blur procedure and runs
on CPU or CUDA. The root README and `docs/REPRODUCIBILITY.md` are the supported
entry points.

The modules here are retained to preserve the original experimental code:

| Modules | Role |
| --- | --- |
| `core.py` | Frozen feature model, signed predicates, pathway selection, image geometry |
| `compare_neuron_cams.py` | Shared target-layer definitions and signed neuron targets |
| `scorecam_all_active.py` | Shared-channel Score-CAM calculation; now respects the model device |
| `render_heatmap_components.py` | All eight-connected component boxes |
| `evaluate_grounding_random100.py` | Historical CUDA Score-CAM experiment and area-matched controls |
| `evaluate_gradcam_random100_four_models.py` | Grad-CAM evaluation using the saved population |
| `extend_scorecam_lower010.py` | Extend saved Score-CAM schedules to cutoff 0.10 |
| `run_demo.py` | Earlier integrated-gradients + LaMa inpainting experiment, **not the current method** |
| `compare_birefnet_isnet_random30.py`, `birefnet_segmenter.py` | Earlier optional foreground-display experiments |

Historical rendering/comparison entry points expect their original result
manifests, which are not shipped. The inpainting experiment additionally requires
local LaMa/ISNet model files. The BiRefNet experiment expects an inspected local
architecture and weights at the revision embedded in its source, plus that
model's own dependencies. These optional experiments are not installed by
`requirements.txt` and are not necessary to explain an image or run the blur
grounding benchmark. Captum is included in `requirements-dev.txt` to preserve
regression tests for the earlier integrated-gradients implementation.

For new benchmark reports, use `benchmark_report.py`; the older four-model report
contains narrative tied to its original run and requires additional audit outputs.
