### Plotting style guide

All new plotting code must use `src.plot_style` for figure creation and axis formatting.

#### Select styles
- Exploratory/interactive:
  - Call `use_exploratory_style()` or `use_default_style()` once per script/notebook.
- Final paper figures:
  - Call `use_paper_style()` once per script that generates publication figures.

#### Create figures
- General figures:
  - `fig, ax = new_figure(kind="single" | "double" | "custom", figsize=...)`
  - Use `nrows`/`ncols` to create subplot grids.
- Radar (polar) figures:
  - `fig, ax, angles = new_radar_figure(labels, kind="single" | "double" | "custom", figsize=...)`
  - Use `setup_radar_axes(ax, labels, title=...)`

#### Axis formatting helpers
- Scatter: `setup_scatter_axes(ax, xlabel=..., ylabel=..., title=...)`
- Bar: `setup_bar_axes(ax, xlabel=..., ylabel=..., title=..., rotate_xticks=True, rotation=30.0)`
- Line: `setup_line_axes(ax, xlabel=..., ylabel=..., title=...)`
- Heatmaps: `setup_heatmap_axes(ax, xlabel=..., ylabel=..., title=..., rotate_xticks=True, rotation=45.0)`
- Violin: `setup_violin_axes(ax, xlabel=..., ylabel=..., title=..., rotate_xticks=True, rotation=30.0)`
- Generic: `setup_axes(ax, xlabel=..., ylabel=..., title=..., grid=True, grid_axis="both")`

Let Seaborn (if used) inherit Matplotlib styles; avoid scattered `sns.set_theme` calls. If a global Seaborn theme is ever needed, add it in `src/plot_style.py` with rationale.

#### Saving figures
- Prefer vector formats (`.pdf` or `.svg`) for final paper figures; `.png` is fine for diagnostics.
- Always include `bbox_inches="tight"`.
  - Example: `fig.savefig("figures/experiment_violin.pdf", bbox_inches="tight")`
- Mark scripts that generate final paper figures and ensure they call `use_paper_style()`.

#### Notes and extensions
- Current helpers target 2D plots and a simple radar plot via Matplotlib polar axes.
- 3D plots (e.g., in `src/visualization.py`) should still create figures via `new_figure(...)`, with axes added using `fig.add_subplot(..., projection="3d")`.
- Future extensions can add:
  - 3D formatting helpers
  - Interactive backends and themes (kept separate from paper style)
  - Centralized color/marker cycles for method families


