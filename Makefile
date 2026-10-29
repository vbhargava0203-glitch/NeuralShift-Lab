PYTHON ?= python
RESULTS ?= results/pilot/pilot

.PHONY: install test lint smoke figures report
install:
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

smoke:
	$(PYTHON) -m neuralshift.cli download-data --config configs/cpu_smoke.yaml
	$(PYTHON) -m neuralshift.cli experiment --config configs/cpu_smoke.yaml

figures:
	$(PYTHON) -m neuralshift.cli figures --results $(RESULTS)

report:
	$(PYTHON) -m neuralshift.cli report --results $(RESULTS)
