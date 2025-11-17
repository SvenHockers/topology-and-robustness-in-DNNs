# topology-and-robustness-in-DNNs
We quantify how the topology of neural activations evolves across layers and test whether these topological trajectories predict model robustness under structured input perturbations.

## Setup and dependencies

This project manages its Python environment with a simple helper script and `requirements.txt`.

### Prerequisites
- Python 3.x installed and available as `python3` (macOS/Linux) or `python` (Windows Git Bash)
- `bash` shell

### First-time installation
```bash
chmod +x manage_dependensies.sh
./manage_dependensies.sh install
```
This will:
- Create a fresh virtual environment in `venv/` (removes an existing one if present)
- Upgrade `pip`
- Install packages from `requirements.txt`

Activate the virtual environment after installation:
- macOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows (Git Bash):
  ```bash
  source venv/Scripts/activate
  ```

### Update dependencies later
```bash
./manage_dependensies.sh update
```
This upgrades `pip` and updates packages to the versions specified in `requirements.txt`.

### Help
```bash
./manage_dependensies.sh --help
```
Also available as `-h` or `help`.

**If you still run into issues contact me (Sven)**

### Notes
- Dependencies are listed in `requirements.txt`.
- The `install` command recreates the `venv/` from scratch; use `update` to keep your existing environment and just refresh packages.

--------------------
## Introduction

This repository studies whether changes in topology inside a network’s representations relate to model robustness. Instead of only tracking accuracy under perturbations, we compute topological summaries of activations layer-by-layer (via persistent homology) and quantify how those summaries change when inputs are perturbed (adversarial or geometric). Intuitively:

- If a model is robust, small input perturbations should not drastically alter the “shape” of its internal representations.
- If a model is brittle, even small perturbations might cause large topological changes at certain layers.

What we measure
- For selected layers and homology dimensions (H0 components, H1 loops), we compute persistence diagrams on activations for clean and perturbed inputs.
- We extract stats (counts, total/mean/max persistence, entropy) and distances between diagrams (e.g., Wasserstein). These serve as “topology sensitivity” metrics per layer and condition.
- We relate these to robustness metrics: robust accuracy (RA@eta) and minimal adversarial radius eta*.

How the pipeline helps a research workflow
- Repeatable: a YAML-configured pipeline generates data, trains/loads a model, runs probes, and writes standardized metrics/plots for comparisons.
- Diagnostic: heatmaps and curves show which layers/topological features change most, at which eta and under which perturbations.
- Comparative: normalized heatmaps subtract a clean–clean “noise floor,” enabling comparisons across layers/models without scale artifacts.
- Actionable: the outputs highlight layers that could benefit from regularization or architectural changes, and quantify the effect across norms and eta.

Audience
- The scripts and configs aim to be accessible to master’s students and research engineers. You can run baseline experiments out-of-the-box and then tune knobs (e.g., normalization, PCA, eta grids) as hypotheses evolve.

## Scripts and how to use them

### 1) Config-driven robustness pipeline: `scripts/run_robustness.py`

```bash
python scripts/run_robustness.py --config configs/robustness/default.yaml
```

CLI arguments:
- `--config PATH` (required): YAML config (see below).
- `--output_dir DIR` (optional): override output root in YAML.
- `--checkpoint PATH` (optional): load an existing model.
- `--exp_name NAME` (optional): override experiment name.
- `--sample_limit N` (optional): cap number of validation samples processed.

What it does:
- Generates a synthetic dataset of point clouds (circles, spheres, tori).
- Trains the selected model (MLP or CNN) or loads a checkpoint.
- Runs robustness probes (adversarial, geometric, interpolation).
- Computes layer-wise topology (persistence diagrams), statistics and distances.
- Writes metrics and plots to a timestamped folder.

Outputs (under `outputs/`):
- `config.yaml`, `resolved_config.json`: config used.
- `versions.json`: package versions used.
- `run.log`: logs.
- `model.pth`: trained weights if training was enabled.
- `metrics.csv`: per-sample summary.
- `layerwise_topology.csv`: per-sample stats per layer).
- `diagram_distances.csv`: per-sample distances per layer and condition, plus `noise_floor` rows.
- `summary.json`: aggregates and robust accuracy curves.
- Plots:
  - Robust accuracy curves: `ra_curve_{linf,l2}.png`
  - Histograms: `hist_eps_{linf,l2}.png`
  - Distance bars: `layer_wasserstein_H{0,1}.png`
  - Heatmaps (raw and normalized by noise floor): `heatmap_wasserstein_H{0,1}.png`, `heatmap_wasserstein_H{0,1}_norm.png`
  - Sensitivity curves (distance vs eta): `curves_wasserstein_H1_{linf,l2}.png`
  - Scatter (eta* vs distance) on best layer/H: `scatter_eps_{linf,l2}_vs_dist_H{H}_{layer}.png`
  - Violin distributions at max eta: `violin_wasserstein_H1_{linf,l2}.png`
  - Sample diagrams by class: `persistence_diagrams_by_class.png`

YAML configuration
- For a fully commented reference of all settings see:
  - `configs/robustness/annotated_template.yaml`

<!-- 
Tips for meaningful results:
- Prefer `normalize: zscore` and `pca_dim: 16` to stabilize TDA across layers.
- Exclude `pooled` for H1 in correlations (loops often vanish after pooling).
- Use non-trivial eta in `layerwise_topology.conditions` (e.g., `0.2, 0.4`) to reveal structure.
- `diagram_distances.csv` includes `noise_floor` to contextualize the effect size; the normalized heatmaps subtract this baseline.
-->

## How the pipeline works

- Flow: Data -> Model -> Probes (adversarial/geometric/interpolation) -> Layerwise TDA -> Reporting.
- Why:
  - TDA on activations (not inputs) captures how the network reshapes geometry under stress.
  - Normalization/PCA stabilize diagrams across layers; subsample/bootstrap balance cost and variance.
  - Noise‑floor (clean–clean) distances provide a baseline; normalized heatmaps show true effect sizes.
- Outputs:
  - `metrics.csv` (eta*, thresholds), `layerwise_topology.csv` (clean stats), `diagram_distances.csv` (clean->perturbed + noise_floor).
  - Plots: bars (per‑layer), heatmaps (raw/normalized), distance‑vs‑eta curves (mean±CI), scatter (eta* vs distance), violins (distributions).
- Key knobs:
  - `probes.topology.normalize: zscore`, `pca_dim: 16` for stable TDA.
  - `probes.layerwise_topology.conditions.adv_linf_eps: [0.2, 0.4]` to avoid trivial zeros.
  - Prefer pre‑pooling layers for H1; H0 often informative across layers.
  - Increase `sample_size` (e.g., 300) if runtime allows for smoother diagrams.
