PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
EXPORTS ?= exports
EXAMPLE_DATA ?= data/demo/demo_cohort.csv
EXAMPLE_HISTORY ?= data/demo/demo_prior_projects.csv
PROJECTS ?= 3

.PHONY: help run demo test test-all lint lint-free cli-help allocate-example docs docs-shipped notices

help:
	@echo "make run               start the Streamlit app"
	@echo "make demo              start the app in demo mode (fictional playful data)"
	@test -d tests && echo "make test              fast pytest suite (slow tests skipped)" || true
	@test -d tests && echo "make test-all          pytest including @pytest.mark.slow tests" || true
	@echo "make lint              ruff correctness gate (unused imports/variables, undefined names)"
	@echo "make lint-free         byte-compile app/ to catch syntax errors"
	@echo "make cli-help          command-line interface help (python -m app.cli)"
	@echo "make allocate-example  CLI allocation of $(EXAMPLE_DATA) after $(EXAMPLE_HISTORY) into $(EXPORTS)/"
	@echo "make docs              list the documentation"

run:
	$(PYTHON) -m streamlit run app/main.py

demo:
	GROUP_DIVERSIFIER_DEMO=1 $(PYTHON) -m streamlit run app/main.py

test:
	@if [ -d tests ]; then $(PYTHON) -m pytest; else echo "make test: this copy has no tests/ folder; nothing to run"; fi

test-all:
	@if [ -d tests ]; then $(PYTHON) -m pytest --run-slow; else echo "make test-all: this copy has no tests/ folder; nothing to run"; fi

lint-free:
	$(PYTHON) -m compileall -q app

lint:
	$(PYTHON) -m ruff check .

cli-help:
	@$(PYTHON) -m app.cli --help
	@for cmd in attributes presets suggest-policy feasibility allocate evaluate compare; do \
		echo; echo "==================== $$cmd ===================="; \
		$(PYTHON) -m app.cli $$cmd --help; \
	done

allocate-example:
	@mkdir -p $(EXPORTS)
	$(PYTHON) -m app.cli allocate --data $(EXAMPLE_DATA) --history $(EXAMPLE_HISTORY) \
		--projects $(PROJECTS) --out $(EXPORTS)/allocation.csv \
		--report $(EXPORTS)/allocation_report.json \
		--report-html $(EXPORTS)/allocation_report.html
	$(PYTHON) -m app.cli evaluate --data $(EXAMPLE_DATA) --allocation $(EXPORTS)/allocation.csv \
		--history $(EXAMPLE_HISTORY) --quiet

docs: docs-shipped

docs-shipped:
	@echo "Documentation:"
	@echo "  README.md                product overview, quick start"
	@test -f USER_GUIDE.md && echo "  USER_GUIDE.md            coordinator workflow through the app" || true
	@echo "  config/policies/README.md  the example policies"

CONTAINER_ENGINE ?= podman
IMAGE ?= group-diversifier
VERSION := $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' app/__init__.py)

.PHONY: container-build container-run lock

container-build:
	$(CONTAINER_ENGINE) build --format docker -t $(IMAGE):$(VERSION) -t $(IMAGE):latest -f Containerfile .

container-run:
	$(CONTAINER_ENGINE) run --rm -p 127.0.0.1:8501:8501 --name group-diversifier $(IMAGE):$(VERSION)

LOCK_FLAGS ?=
LOCK_COMMON = --python-version 3.12 --universal --generate-hashes $(LOCK_FLAGS)

lock:
	uv pip compile requirements.txt $(LOCK_COMMON) -o requirements.lock.txt
	uv pip compile requirements-dev.in -c requirements.lock.txt $(LOCK_COMMON) -o requirements-dev.lock.txt

notices:
	$(PYTHON) scripts/make_third_party_notices.py
