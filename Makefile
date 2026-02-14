.PHONY: install check-env train eval deploy lint test clean

# Default Python — override with: make train PYTHON=python3.11
PYTHON ?= python
CONFIG ?= configs/train_ppo.yaml

install:
	$(PYTHON) -m pip install -e ".[dev]"

check-env:
	$(PYTHON) -m scripts.check_env

train:
	$(PYTHON) -m src.training.train --config $(CONFIG)

eval:
	$(PYTHON) -m src.evaluation.evaluate --model $(MODEL) --config $(CONFIG)

deploy:
	$(PYTHON) -m src.deployment.deploy --model $(MODEL) --config $(CONFIG)

lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
