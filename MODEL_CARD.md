# Model card

## Status

This repository implements and tests an educational Vision Transformer from
scratch. No full-dataset GPU training, three-seed benchmark, or trained-weight
release has been completed in the cloud development task. Synthetic runs verify
infrastructure only. Consult `reports/VALIDATION.md` for measured CPU checks.

## Architecture and datasets

ViT: explicit Q/K/V projections, scaled dot-product attention, pre-LayerNorm,
GELU MLP, residual paths, stochastic depth; CLS or patch-average pooling.
The default full configuration uses 32x32 images, 4x4 patches, dimension 128,
6 blocks, 4 heads and 10 classes. CIFAR-10 is the primary intended benchmark;
CIFAR-100 is a configured extension. The torchvision data loader checks the
upstream archive's MD5; TLS verification remains enabled. The model does not
use timm internals. timm 1.0.15 is a correctness/baseline reference only.

CIFAR-10/CIFAR-100 are third-party research datasets; this code's MIT license
is not a license for those data. Consult the dataset publisher's terms.
A fixed stratified 5,000-image validation set comes from the original training
set; the official test set is accessed only for explicit final reporting.
No Tiny-ImageNet loader or experiment is implemented: it is optional and is
not natively provided by torchvision as a dedicated dataset class.

## Intended use

Learning tensor operations, reproducible small-image classification experiments,
and evaluating architecture/augmentation tradeoffs. Not suitable for medical,
safety-critical, identity, or deployment decisions without independent testing.

## Limitations

Small-data ViTs lack the spatial locality/translation biases of CNNs and often
need more data or regularization. That is motivation, not a measured result
for this repository. The small ResNet baseline has a different parameter count;
matching epochs, optimizer and augmentation does not match compute or capacity.
Report both counts and measured GPU times. Hyperparameters are not claimed optimal.

Attention is not a causal explanation. Rollout omits value projections/MLPs and
averages heads. FLOP reports omit unsupported operators and count one multiply-add
as one FLOP. Mixed precision and CUDA execution require GPU validation.
Bitwise epoch-boundary CPU resume is tested; reproducibility across hardware,
PyTorch versions, or a mid-epoch crash is not promised. No distributed training.

Checkpoints are pickle-based; never load untrusted checkpoint files.

## Results and releases

No headline accuracy is reported until at least three completed real-data seeds
and explicit final-test evaluations exist. Report mean and sample standard
deviation, the split fingerprint, config, code revision and compute budget.
An interrupted/debug/subset/synthetic run is not a benchmark result.

When a final model is validated, publish `best.pt`, its resolved YAML, metrics,
SHA-256 checksum and this card as a GitHub Release asset. Do not commit weights,
data or TensorBoard logs. No trained weights are currently published.
