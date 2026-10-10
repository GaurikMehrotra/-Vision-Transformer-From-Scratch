# Measured validation record

Validated in the CPU cloud environment on 2026-10-10 using Python 3.12.14,
PyTorch 2.6.0+cpu, Torchvision 0.21.0+cpu and timm 1.0.15.
CUDA is unavailable. This is development validation, not benchmark training.

| Check | Observed outcome |
|---|---|
| Full pytest | 55 passed |
| Fast CPU CI subset | 51 passed, 4 deselected |
| Ruff lint and format | Passed; 33 Python files formatted |
| Dependency consistency | `pip check` passed |
| Conv vs unfold | Output, input gradient and weight gradient match |
| Explicit attention vs PyTorch MHA | Output, head weights and input gradients match |
| Full ViT vs timm | Four learned/none × CLS/average configurations pass |
| Every parameter's gradient | Finite for both patches, all positions and pools |
| Single-batch learning | Cross entropy below 0.01 and all four labels correct |
| Determinism | Seeded initialization/training dropout and eval repeat pass |
| CPU resume | Model and AdamW states bitwise equal to uninterrupted training |
| Synthetic CLI train | Two epochs complete; checkpoint/config/split/CSV/JSON/events saved |
| Synthetic CLI evaluate | Validation request completed, 16 examples |
| Synthetic CLI visualization | Head maps, rollout, overlay, position similarity and patch filters saved |
| Synthetic augmentation preview | PNG saved |
| Baseline | ResNet forward and synthetic train/evaluate pass |
| Experiment planning | Six ViT/ResNet seed runs planned without executing GPU training |
| Aggregation | Synthetic excluded; duplicate/provenance checks and mean/sample std tested |
| Notebook walkthrough | Executed all component cells, ending at [2,10] logits |
| Real CIFAR debug | Blocked: official archive request received proxy `403 Forbidden` |
| GPU AMP and GPU resume | Unrun: no CUDA device |
| Real-data baseline, ablations, three seeds | Unrun |
| CIFAR-100 | Configured, unrun |
| Tiny-ImageNet | Optional, not implemented |
| Real-trained interpretation and final test | Unrun |
| Trained-weight release | Not published |

Matplotlib/pyparsing deprecation warnings and Python multiprocessing fork warnings
were observed; checks completed. No test assertions were disabled or failures
converted to skips. Four slow tests are intentionally deselected in the CI subset,
including single-batch learning, resume, baseline smoke and figure artifacts.
The full suite executes them all.

## Parameter and counted-FLOP evidence

| Configuration | Parameters | Counted FLOPs/image |
|---|---:|---:|
| Debug ViT (D64, depth2, heads4, P4, CLS) | 108,106 | 7,772,608 |
| Default ViT (D128, depth6, heads4, P4, CLS) | 1,205,898 | 84,102,016 |
| ResNet-20 (width16) | 272,474 | 41,214,592 |

Both ViT counts exactly match their timm references. These are architecture
profiles of untrained models. One multiply-add is one counted FLOP. Elementwise
add/multiply, softmax, GELU and the ResNet mean are omitted as listed in the three
JSON profiles. Do not interpret the numbers as measured runtime or full arithmetic.

## Outstanding execution

The required CIFAR hostname `www.cs.toronto.edu` was saved as a custom domain in
the environment configuration draft. Saving does not apply/publish that change.
Review and save it in environment settings; an official redirect hostname may also
need adding when observed. Keep TLS and upstream checksum verification enabled.
Alternatively use your own verified CIFAR archive at the configured data root.

Use `docs/KAGGLE.md` to execute first-epoch GPU/AMP/resume checks, full training,
matched-budget baseline, three seeds and selected ablations. Select configurations
only with validation, then evaluate the official test set explicitly. Update this
record and the README using actual results; do not label synthetic accuracy or
reference-model correctness as trained benchmark performance.
