.PHONY: help install test cli-help \
	ood-synthetic-shapes \
	run-tabular run-mnist run-synthetic-shapes \
	run-blobs run-nested-spheres run-torus-one-hole run-torus-two-holes \
	run-all-final

PYTHON ?= python3
SPACE ?= optimisers/spaces/constrains.yaml
OUT ?= out

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  test      Run lightweight smoke tests"
	@echo "  cli-help  Show help for the optimiser CLI"
	@echo "  ood-synthetic-shapes  Run ONLY OOD configs for synthetic_shapes (batch optimiser)"
	@echo "  run-tabular            Run ALL configs under config/final/tabular"
	@echo "  run-mnist              Run ALL configs under config/final/mnist"
	@echo "  run-synthetic-shapes   Run ALL configs under config/final/synthetic_shapes"
	@echo "  run-blobs              Run ALL configs under config/final/blobs"
	@echo "  run-nested-spheres     Run ALL configs under config/final/nested_spheres"
	@echo "  run-torus-one-hole     Run ALL configs under config/final/torus_one_hole"
	@echo "  run-torus-two-holes    Run ALL configs under config/final/torus_two_holes"
	@echo "  run-all-final          Run all datasets under config/final/* (sequential)"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests

cli-help:
	$(PYTHON) -m optimisers --help
	$(PYTHON) -m optimisers batch --help

# Run ONLY synthetic_shapes OOD configs (skip baseline/ + base_*.yaml).
#
# Override defaults if desired, e.g.:
#   make ood-synthetic-shapes OOD_DATASET=synthetic_shapes_3class OOD_TRIALS=5
OOD_CONFIG_DIR ?= config/final/synthetic_shapes
OOD_DATASET ?= synthetic_shapes_2class
OOD_MODEL ?= CNN
OOD_TRIALS ?= 15
OOD_INITIAL ?= 5
OOD_SEED ?= 30

ood-synthetic-shapes:
	$(PYTHON) -m optimisers batch \
	  --config-dir $(OOD_CONFIG_DIR) \
	  --dataset-name $(OOD_DATASET) \
	  --model-name $(OOD_MODEL) \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --ignore-baseline \
	  --ignore "base*" \
	  --n-trials $(OOD_TRIALS) \
	  --n-initial $(OOD_INITIAL) \
	  --seed $(OOD_SEED)

# Full dataset batch runs (no ignore flags; runs everything under each config/final/<dataset>/ directory)
RUN_TRIALS ?= 15
RUN_INITIAL ?= 5
RUN_SEED ?= 30

SYNTH_DATASET ?= synthetic_shapes_2class
SYNTH_MODEL ?= CNN

run-tabular:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/tabular \
	  --dataset-name TABULAR \
	  --model-name MLP \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

run-mnist:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/mnist \
	  --dataset-name IMAGE \
	  --model-name CNN \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

run-synthetic-shapes:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/synthetic_shapes \
	  --dataset-name $(SYNTH_DATASET) \
	  --model-name $(SYNTH_MODEL) \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

# Pointcloud datasets use the VECTOR dataset loader; dataset_type is set inside each YAML.
run-blobs:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/blobs \
	  --dataset-name VECTOR \
	  --model-name MLP \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

run-nested-spheres:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/nested_spheres \
	  --dataset-name VECTOR \
	  --model-name MLP \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

run-torus-one-hole:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/torus_one_hole \
	  --dataset-name VECTOR \
	  --model-name MLP \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

run-torus-two-holes:
	$(PYTHON) -m optimisers batch \
	  --config-dir config/final/torus_two_holes \
	  --dataset-name VECTOR \
	  --model-name MLP \
	  --space $(SPACE) \
	  --metric-path auto \
	  --output-root $(OUT) \
	  --n-trials $(RUN_TRIALS) \
	  --n-initial $(RUN_INITIAL) \
	  --seed $(RUN_SEED) \
	  --make-plots

run-all-final: run-tabular run-mnist run-synthetic-shapes run-blobs run-nested-spheres run-torus-one-hole run-torus-two-holes
