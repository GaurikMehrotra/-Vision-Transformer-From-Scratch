# Phase guide and actual completion status

The complete source for each phase is in the linked repository files; no model
implementation is hidden in notebooks. The initial project is delivered in one
validated implementation commit. The commit commands below are useful if following
the phases on your own learning branch; on the delivered branch the files are
already committed and those commits need not be repeated.

## 1 — LOCAL: environment, configuration, seed, token walkthrough

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest tests/test_artifacts.py::test_walkthrough_notebook_executes -q
git add pyproject.toml requirements.txt src/__init__.py src/utils.py configs notebooks/01_components_walkthrough.ipynb .gitignore
git commit -m "Prepare CPU environment, configs and image-to-token walkthrough"
```

See `src/utils.py`, `configs/debug_cpu.yaml`, and notebook 01. Expect image
`[2,3,32,32]`, patch grid `[2,64,8,8]`, tokens `[2,64,64]`, CLS tokens
`[2,65,64]`, attention `[2,4,65,65]`, logits `[2,10]`. Implemented and CPU-verified.

## 2 — LOCAL: patch embeddings

```bash
python -m pytest tests/test_patch_equiv.py -q
git add src/models/patch_embed.py tests/test_patch_equiv.py
git commit -m "Implement equivalent Conv2d and unfold patch projections"
```

Complete code: `src/models/patch_embed.py`. Test copies flattened conv weights and
checks outputs and input/weight gradients in float64. Implemented and verified.

## 3 — LOCAL: four position encodings

```bash
python -m pytest tests/test_shapes.py -q -k position
git add src/models/pos_encoding.py tests/test_shapes.py
git commit -m "Add learned, sinusoidal, 2D sin-cos and no-position encodings"
```

Complete code: `src/models/pos_encoding.py`. Shape and axis-coordinate tests pass.

## 4 — LOCAL: explicit multi-head attention

```bash
python -m pytest tests/test_attention_equiv.py -q
git add src/models/attention.py tests/test_attention_equiv.py notebooks/01_components_walkthrough.ipynb
git commit -m "Implement and independently verify explicit multi-head attention"
```

Complete code: `src/models/attention.py`. Copied weights match PyTorch MHA outputs,
per-head probabilities and input gradients. Float64 `atol=1e-10, rtol=1e-9`.

## 5 — LOCAL: MLP, DropPath, pre-LN block

```bash
python -m pytest tests/test_shapes.py -q -k "modules or droppath"
git add src/models/mlp.py src/models/droppath.py src/models/block.py tests/test_shapes.py
git commit -m "Add GELU MLP, sample-wise stochastic depth and residual blocks"
```

Shape/mask/expectation tests pass. Complete code is in the three named files.

## 6 — LOCAL: ViT assembly, gradients, profile

```bash
python -m pytest tests/test_gradients.py tests/test_shapes.py -q
python -m src.profile --config configs/debug_cpu.yaml --compare-timm
git add src/models/vit.py src/profile.py tests/test_gradients.py reports/profile_debug_cpu.json
git commit -m "Assemble CLS and average-pool ViTs with initialization and profiling"
```

All parameters receive finite gradients across both patch methods, four position
strategies and both pools. Debug counts: 108,106 parameters, 7,772,608 recognized
FLOPs/image. Unsupported ops are disclosed; counts equal timm.

## 7 — LOCAL: timm full-model equivalence

```bash
python -m pytest tests/test_timm_equiv.py -q
git add tests/test_timm_equiv.py src/profile.py
git commit -m "Verify copied timm ViT logits and matched parameter/FLOP counts"
```

Four combinations of pool/position pass at `atol=1e-5, rtol=1e-4`. See README for
the explicit zero-position-buffer handling of timm's no-position CLS behavior.

## 8 — LOCAL: data and augmentation

```bash
python -m pytest tests/test_data.py -q
python -m src.data.preview --config configs/debug_cpu.yaml
git add src/data tests/test_data.py
git commit -m "Add held-out CIFAR splits and from-scratch Mixup/CutMix"
```

Synthetic/algorithm tests pass. Actual CIFAR preview/debug needs the official
archive; cloud download is blocked by proxy 403. No real-data success claimed.

## 9 — LOCAL debug, then KAGGLE: training and resume

```bash
python -m pytest tests/test_overfit.py tests/test_training.py -q
python -m src.train --config configs/smoke_synthetic.yaml
python -m src.train --config configs/debug_cpu.yaml
git add src/train.py src/evaluate.py tests/test_overfit.py tests/test_training.py configs/debug_cpu.yaml
git commit -m "Add resumable AdamW training, validation selection and logging"
```

Single-batch loss falls below 0.01. Uninterrupted and resumed CPU training have
bitwise-equal model/optimizer states. Synthetic two-epoch training passes.
Actual CIFAR debug and first GPU training are pending; exact cells are in
`docs/KAGGLE.md`, cells 1–4. Model code remains unchanged across CPU/CUDA.

## 10 — KAGGLE: ResNet baseline

Use `docs/KAGGLE.md` cell 5. Before GPU execution the config/model files can be
committed with:

```bash
git add src/baselines/resnet_small.py configs/resnet_cifar10.yaml
git commit -m "Add matched-budget ResNet-20 baseline configuration"
```

Implementation and CPU forward test pass. Real baseline training is not run.

## 11 — KAGGLE: ablations and three seeds

Use cells 5–6 and 8 in `docs/KAGGLE.md`. Planning and aggregation commands:

```bash
python -m src.experiments --configs configs/vit_tiny_cifar10.yaml configs/resnet_cifar10.yaml --seeds 42 43 44
python -m src.results --root runs/experiments --output reports/generated/results
git add src/experiments.py src/results.py configs/ablations
git commit -m "Add resumable multi-seed experiment planning and honest result tables"
```

Expect a six-run plan; no results until actual GPU runs finish. One-seed ablations
are exploratory. No headline, ablation finding, or measured superiority claimed.

## 12 — LOCAL/KAGGLE: interpretation

```bash
python -m src.visualize --checkpoint runs/smoke_synthetic/best.pt --output reports/generated/smoke_figures
python -m pytest tests/test_artifacts.py -q
git add src/visualize.py tests/test_artifacts.py notebooks/02_attention_viz.ipynb
git commit -m "Export attention heads, rollout, positions and patch-filter figures"
```

Artifact generation and rollout normalization/order are verified. Inspect real
trained outputs using Kaggle cell 9; empirical analysis remains pending training.

## 13 — KAGGLE, optional: CIFAR-100

Use Kaggle cell 7. CIFAR-100 config is provided. Tiny-ImageNet is not implemented;
it is optional and requires a separately verified ImageFolder adapter.

```bash
git add configs/vit_tiny_cifar100.yaml
git commit -m "Provide optional CIFAR-100 experiment configuration"
```

No CIFAR-100 or Tiny-ImageNet run is claimed.

## 14 — LOCAL: quality and CPU CI

```bash
make lint
make test
make test-fast
git add tests Makefile .github/workflows/ci.yml .gitignore LICENSE
git commit -m "Add CPU correctness CI, quality checks and artifact exclusions"
```

Local lint/full/fast checks are run. GitHub Actions is configured; its remote
execution must be checked after push and is not established by local success.

## 15 — LOCAL: honest documentation and weights plan

```bash
git add README.md MODEL_CARD.md docs notebooks/03_results.ipynb reports/VALIDATION.md
git commit -m "Document tensor shapes, GPU reproduction, status and model limitations"
```

Docs describe actual CPU verification and absent benchmark results. Results,
trained attention figures and a final weight release remain gated on training.
See `docs/RELEASE.md`; no untrained weight is published as a trained model.

## 16 — LOCAL: final checks and GitHub

```bash
make lint
make test
git status --short
git push -u origin main
```

The existing GitHub repo is the target. `gh repo create` instructions for a new
fork, final-test evaluation and checkpoint Release commands are in
`docs/RELEASE.md`. Code push is separate from GPU completion and weight publishing.
