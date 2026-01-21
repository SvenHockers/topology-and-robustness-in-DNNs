.PHONY: help install test cli-help \
	run-tabular run-mnist run-synthetic-shapes \
	run-blobs run-nested-spheres run-torus-one-hole run-torus-two-holes \
	run-all \
	post-analyses

PYTHON ?= python3
SPACE ?= optimisers/spaces/constrains.yaml
OUT ?= out

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  test      Run lightweight smoke tests"
	@echo "  cli-help  Show help for the optimiser CLI"
	@echo "  run-tabular            Run ALL configs under config/final/tabular"
	@echo "  run-mnist              Run ALL configs under config/final/mnist"
	@echo "  run-synthetic-shapes   Run ALL configs under config/final/synthetic_shapes"
	@echo "  run-blobs              Run ALL configs under config/final/blobs"
	@echo "  run-nested-spheres     Run ALL configs under config/final/nested_spheres"
	@echo "  run-torus-one-hole     Run ALL configs under config/final/torus_one_hole"
	@echo "  run-torus-two-holes    Run ALL configs under config/final/torus_two_holes"
	@echo "  run-all                Run all datasets under config/final/* (sequential)"
	@echo "  post-info-gain         Post-analysis: info gain / residual-topology plots"
	@echo "  post-logres-rocs       Post-analysis: Mahalanobis vs LogReg ROC curves"
	@echo "  post-analyses          Run all post-analyses (info gain + ROC curves)"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests

cli-help:
	$(PYTHON) -m optimisers --help
	$(PYTHON) -m optimisers batch --help

RUN_TRIALS ?= 10
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

run-all: run-tabular run-mnist run-synthetic-shapes run-blobs run-nested-spheres run-torus-one-hole run-torus-two-holes

post-info-gain:
	$(PYTHON) post_analyses/calculate_info_gain.py \
	  --out-root $(OUT) \
	  --save-dir $(OUT)/_analysis/knn_vs_topo_adv

post-logres-rocs:
	$(PYTHON) post_analyses/generate_logres_roc_curves.py \
	  --out-root $(OUT) \
	  --save-dir $(OUT)/_analysis/topo_upper_bound_rocs

post-analyses: post-info-gain post-logres-rocs