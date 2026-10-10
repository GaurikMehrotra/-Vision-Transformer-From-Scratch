# GPU execution and interruption recovery

These are executable notebook cells, not evidence that GPU experiments ran.
Choose a GPU accelerator in Kaggle settings and enable Internet. Use one GPU:
this project does not implement DDP. Free GPU quotas and session durations vary.
Run a short smoke/first epoch, measure its time, then estimate the remaining
budget; do not assume all ablations fit in 30 hours. Reduce the **shared** budget
before beginning comparisons if needed, and record it in new config files.

## Cell 1 — clone GitHub

```python
import subprocess
from pathlib import Path
REPO = Path('/kaggle/working/vit-from-scratch')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/GaurikMehrotra/-Vision-Transformer-From-Scratch.git', str(REPO)], check=True)
%cd /kaggle/working/vit-from-scratch
!git rev-parse HEAD
```

For Colab, replace `/kaggle/working` with `/content` consistently and use a persistent
Drive directory for output. Avoid opening multiple notebooks writing to one run.

## Cell 2 — paired CUDA dependencies

```python
import sys, subprocess
# Official PyTorch 2.6 / torchvision 0.21 CUDA 12.4 pair. Verify the current
# accelerator driver can support this CUDA runtime; never disable signatures/TLS.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--index-url',
               'https://download.pytorch.org/whl/cu124',
               'torch==2.6.0', 'torchvision==0.21.0'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps', '--no-build-isolation'], check=True)
```

If Torch was already imported, restart the kernel after installation, return to
the checkout, and rerun the following cell. Select a runtime offered on the
[official PyTorch previous-version page](https://pytorch.org/get-started/previous-versions/)
if the available driver cannot support CUDA 12.4. Keep the Torch/Torchvision pair.
Record the chosen wheel/runtime; only CPU PyTorch 2.6 has been tested here.

## Cell 3 — prerequisites and correctness gate

```python
import torch, subprocess, sys
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'Enable a GPU accelerator before real training'
print(torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
```

The full suite includes the overfit and CPU resume checks. Direct real-data
`src.train` commands also run this gate automatically. GPU AMP/resume are not
established merely because CPU correctness passed; check the first GPU run's
finite losses, validation and resume before launching the full experiment plan.

## Cell 4 — first CIFAR-10 run

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'src.train', '--config',
               'configs/vit_tiny_cifar10.yaml', '--output',
               '/kaggle/working/vit-results/vit_tiny_cifar10/seed42',
               '--seed', '42', '--stop-after-epoch', '1'], check=True)
```

This executes one epoch of the original 100-epoch schedule. Confirm `last.pt`,
`best.pt`, metrics, split, resolved config and TensorBoard files exist. Resume
the original planned schedule in the same directory:

```python
subprocess.run([sys.executable, '-m', 'src.train', '--config',
               'configs/vit_tiny_cifar10.yaml', '--output',
               '/kaggle/working/vit-results/vit_tiny_cifar10/seed42', '--seed', '42',
               '--resume', '/kaggle/working/vit-results/vit_tiny_cifar10/seed42/last.pt'], check=True)
```

A changed epoch count/learning rate is rejected on resume. Plan the complete
budget up front. After a disconnect, restore your saved output folder from a
Kaggle Dataset or downloaded archive before using `--resume`. Runtime storage
alone is not durable across abandoned sessions. Use only your own trusted
checkpoints. The checkpoint captures the original split fingerprint and all RNG
states. Mid-epoch progress is replayed from the last completed epoch.

## Cell 5 — headline ViT + ResNet, three seeds

```python
subprocess.run([sys.executable, '-m', 'src.experiments', '--configs',
               'configs/vit_tiny_cifar10.yaml', 'configs/resnet_cifar10.yaml',
               '--seeds', '42', '43', '44', '--output',
               '/kaggle/working/vit-results', '--run'], check=True)
```

The runner skips completed runs and resumes interrupted ones. ViT and ResNet
use the same dataset/split, 100 epochs, batch size, AdamW schedule, weight-decay
policy, label smoothing, and spatial/batch augmentation. Stochastic depth is
ViT-specific; ResNet has BatchNorm and its spatial inductive bias. Parameters,
FLOPs and actual elapsed times differ and must be reported.

## Cell 6 — ablations (start with an affordable subset)

```python
from pathlib import Path
configs = sorted(str(path) for path in Path('configs/ablations').glob('*.yaml'))
# Plan first: this can be expensive. Run selected configs, one seed for exploration.
subprocess.run([sys.executable, '-m', 'src.experiments', '--configs', *configs,
               '--seeds', '42', '--output', '/kaggle/working/vit-ablations'], check=True)
# Explicit execution after measuring cost; select a smaller list if necessary.
# `--run` is intentionally absent from the planning command above.
```

To execute, use the same command with `--run`; repeat promising comparisons
with seeds 42,43,44 before headline claims. One-seed ablations are exploratory.
`no_mixup` and `no_cutmix` preserve the other method's batch frequency; disabled
branches are identity. Dimension/depth/patch changes alter capacity/compute.
The reference has zero ordinary dropout, so `dropout01` adds it and
`no_dropout_from01.yaml` removes it from that alternative, not from the reference.

## Cell 7 — optional CIFAR-100

```python
subprocess.run([sys.executable, '-m', 'src.experiments', '--configs',
               'configs/vit_tiny_cifar100.yaml', '--seeds', '42', '43', '44',
               '--output', '/kaggle/working/vit-cifar100', '--run'], check=True)
```

Tiny-ImageNet is not implemented and is not a native torchvision dataset. It
needs a separately verified ImageFolder adapter and budget; do not claim a run.

## Cell 8 — validation comparisons, then freeze headline models

```python
subprocess.run([sys.executable, '-m', 'src.results', '--root',
               '/kaggle/working/vit-results', '--output',
               '/kaggle/working/vit-results/validation-report'], check=True)
```

Use validation only to choose configurations. Once configurations/seeds and
reporting decisions are frozen, run the official test set for final reporting:

```python
for name in ['vit_tiny_cifar10', 'resnet_cifar10']:
    for seed in [42, 43, 44]:
        run = Path('/kaggle/working/vit-results') / name / f'seed{seed}'
        subprocess.run([sys.executable, '-m', 'src.evaluate', '--checkpoint',
                       str(run / 'best.pt'), '--split', 'test', '--final-test',
                       '--output', str(run / 'test.json')], check=True)
subprocess.run([sys.executable, '-m', 'src.results', '--root',
               '/kaggle/working/vit-results', '--metric', 'test', '--output',
               '/kaggle/working/vit-results/test-report'], check=True)
```

`test.json` must match the run's config fingerprint and selected best epoch.
Do not use final test scores to choose more hyperparameters.

## Cell 9 — interpretability from a trained checkpoint

```python
subprocess.run([sys.executable, '-m', 'src.visualize', '--checkpoint',
               '/kaggle/working/vit-results/vit_tiny_cifar10/seed42/best.pt',
               '--index', '0', '--output', '/kaggle/working/vit-results/figures'], check=True)
```

Repeat indices, including wrong predictions. Read the generated interpretation
notes; write an evidence-based analysis after inspecting real-trained figures.

## Cell 10 — download artifacts

```python
import shutil
from IPython.display import FileLink, display
archive = shutil.make_archive('/kaggle/working/vit-results', 'zip',
                              '/kaggle/working/vit-results')
display(FileLink(archive))
```

Use Kaggle's notebook output **Save Version** or upload your own outputs as a
private Kaggle Dataset before ending a session. An in-process epoch checkpoint
cannot prevent loss of an ephemeral filesystem. On Colab, save to persistent
Drive and use `google.colab.files.download` for the archive. Never commit data,
weights or raw logs to GitHub. Publish chosen weights as Release assets only
after checks and actual training; see `docs/RELEASE.md`.
