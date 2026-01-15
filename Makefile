.PHONY: help install test cli-help

PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  test      Run lightweight smoke tests"
	@echo "  cli-help  Show help for the two stable CLIs"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests

cli-help:
	$(PYTHON) -m optimizers.cli --help
	$(PYTHON) -m optimisers.cli_batch --help

