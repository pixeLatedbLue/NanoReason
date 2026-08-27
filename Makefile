
.DEFAULT_GOAL := help

PYTHON ?= python
IMAGE  ?= nanoreason-serve:local
PORT   ?= 8000

.PHONY: help install test lint serve docker-build docker-run smoke

help:
	@echo "NanoReason make targets:"
	@echo "  make install       Install the package with the serve + dev extras"
	@echo "  make test          Run the unit test suite (no GPU needed)"
	@echo "  make lint          Run ruff over src and tests"
	@echo "  make serve         Start the web app via the nanoreason-serve script"
	@echo "  make docker-build  Build the CPU-only serving image ($(IMAGE))"
	@echo "  make docker-run    Run that image on port $(PORT), results mounted read-only"
	@echo "  make smoke         CPU plumbing check using configs/smoke.toml"

install:
	$(PYTHON) -m pip install -e ".[serve,dev]"

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m ruff check src tests

serve:
	nanoreason-serve

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 -v "$(CURDIR)/results:/app/results:ro" $(IMAGE)

smoke:
	$(PYTHON) -m nanoreason.evaluate --config configs/smoke.toml --run-name smoke --tasks strategyqa
