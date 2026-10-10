PYTHON ?= python

.PHONY: install format lint test test-fast smoke debug profile plan results
install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps --no-build-isolation
format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests
lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests
test:
	$(PYTHON) -m pytest -q
test-fast:
	$(PYTHON) -m pytest -q -m "not slow"
smoke:
	$(PYTHON) -m src.train --config configs/smoke_synthetic.yaml
debug:
	$(PYTHON) -m src.train --config configs/debug_cpu.yaml
profile:
	$(PYTHON) -m src.profile --config configs/debug_cpu.yaml --compare-timm
plan:
	$(PYTHON) -m src.experiments --configs configs/vit_tiny_cifar10.yaml configs/resnet_cifar10.yaml --seeds 42 43 44
results:
	$(PYTHON) -m src.results --root runs/experiments --output reports/generated/results
