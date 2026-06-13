.PHONY: help install install-dev lint format type test cov train eval serve dashboard docker repro clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:           ## Install runtime package
	pip install -e .

install-dev:       ## Install with all extras + dev tooling + pre-commit hooks
	pip install -e ".[all]"
	pre-commit install

lint:              ## Ruff lint
	ruff check src tests

format:            ## Ruff autoformat + import sort
	ruff format src tests
	ruff check --fix src tests

type:              ## Static type-check
	mypy src

test:              ## Run fast unit tests
	pytest -m "not slow and not gpu"

cov:               ## Full test suite with coverage
	pytest

train:             ## Train (override cfg from CLI, e.g. make train ARGS="model=conformer")
	eegpipe-train $(ARGS)

eval:              ## Evaluate a checkpoint
	eegpipe-eval $(ARGS)

serve:             ## Launch the FastAPI inference server
	uvicorn eegpipe.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:         ## Launch the Streamlit monitoring dashboard
	streamlit run src/eegpipe/dashboard/app.py

repro:             ## Reproduce the DVC pipeline end-to-end
	dvc repro

docker:            ## Build the runtime image
	docker build -f docker/Dockerfile -t eegpipe:latest .

clean:             ## Remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# ---- Phase 5: export & regression gate ----
CUR ?= reports/sleep_metrics.json
METRIC ?= test/macro_f1
TOL ?= 0.02

export:            ## Export a checkpoint to ONNX (ARGS="model.ckpt model.onnx --kind sequence --channels 2 --times 3000 --seq-len 20")
	eegpipe-export $(ARGS)

regression:        ## Fail if $(CUR) regressed vs tests/baseline_metrics.json on $(METRIC)
	python scripts/check_regression.py --baseline tests/baseline_metrics.json --current $(CUR) --metric $(METRIC) --tol $(TOL)

demo:              ## End-to-end smoke test (synthetic; `make demo REAL=--real` for Sleep-EDF)
	python scripts/smoke_demo.py $(REAL)
