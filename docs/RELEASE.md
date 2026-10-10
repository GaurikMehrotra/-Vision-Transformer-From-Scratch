# Publishing code and final weights

This repository already has a GitHub origin. Create a new GitHub repository only
for your own fork; do not run `gh repo create` over the existing remote.

```bash
git status --short
make lint
make test
git add src tests configs notebooks docs reports README.md MODEL_CARD.md Makefile requirements.txt pyproject.toml .gitignore LICENSE .github
git commit -m "Implement from-scratch ViT with correctness tests and reproducible training"
git push -u origin main
```

For a new independent repository, with GitHub CLI authentication already set up:

```bash
gh repo create vit-from-scratch --public --source=. --remote=portfolio --push
```

Full-data GPU experiments are prerequisites for a final trained-model release.
Freeze the chosen configuration using validation only, run the final test set,
report at least three seeds' mean and sample standard deviation, and update the
model card and README with real measurements. A representative released seed
must be named explicitly; never cherry-pick the best test seed without disclosure.

```bash
mkdir -p release
cp runs/experiments/vit_tiny_cifar10/seed42/best.pt release/vit-cifar10-seed42.pt
cp runs/experiments/vit_tiny_cifar10/seed42/config.yaml release/config.yaml
cp runs/experiments/vit_tiny_cifar10/seed42/test.json release/test.json
cp MODEL_CARD.md release/MODEL_CARD.md
sha256sum release/vit-cifar10-seed42.pt > release/SHA256SUMS
# Choose a new tag only after confirming the benchmark and report are complete.
gh release create v1.0.0 release/vit-cifar10-seed42.pt release/config.yaml release/test.json release/MODEL_CARD.md release/SHA256SUMS --title "CIFAR-10 ViT: reproducible three-seed benchmark" --notes-file reports/RELEASE_NOTES.md
```

`reports/RELEASE_NOTES.md` must be written from actual results before that final
command; there is intentionally no fake release-note file or untrained weight
asset. Do not execute a trained-model release for the synthetic smoke checkpoint.
Checkpoints use pickle; verify checksums, provenance and trust before loading.
