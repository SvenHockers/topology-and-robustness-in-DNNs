from __future__ import annotations

from typing import Iterable, Tuple, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import lovelyplots  # noqa: F401  ensure styles are registered; fail fast if unavailable
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib import colormaps as mpl_cmaps
plt.rcParams["axes.formatter.use_mathtext"] = True


# Base styles; adjust if needed.
DEFAULT_STYLES: list[str] = [
	"ipynb",           # base scientific layout
	"colorsblind10",   # colorblind-safe palette (from lovelyplots)
]

PAPER_STYLES: list[str] = [
	"paper",
	"colorsblind10",
]

EXPLORATORY_STYLES: list[str] = [
	"ipynb",
	"colorsblind10",
]

# Default colormaps for heatmaps
HEATMAP_CMAP_SEQUENTIAL: str = "cividis"
HEATMAP_CMAP_DIVERGING: str = "RdBu_r"
PROJECT_HEATMAP_SEQ_NAME: str = "project_heatmap_seq"

# Recommended figure sizes for paper (in inches).
SINGLE_COL_SIZE: Tuple[float, float] = (3.4, 2.5)
DOUBLE_COL_SIZE: Tuple[float, float] = (7.0, 3.0)

def _get_color_cycle() -> list[str]:
	"""Return the current Matplotlib color cycle as a list of hex/rgb strings."""
	prop = plt.rcParams.get("axes.prop_cycle")
	if not prop:
		return []
	return list(prop.by_key().get("color", []))


def _make_sequential_from_color(color: str, name: str) -> LinearSegmentedColormap:
	"""
	Create a light-to-saturated sequential colormap derived from a base color.
	Starts near white and ramps to the base color for perceptual consistency with line colors.
	"""
	base = np.array(to_rgb(color), dtype=float)
	# Build gradient from near-white to base color
	t_vals = np.linspace(0.05, 1.0, 256)  # avoid pure white at the very start
	colors = [(1 - t) * np.ones(3) + t * base for t in t_vals]
	return LinearSegmentedColormap.from_list(name, colors)


def _apply_project_cmaps() -> None:
	"""
	Register and set the project's default heatmap colormap so heatmaps align
	visually with the first color in the current color cycle.
	"""
	colors = _get_color_cycle()
	base = colors[0] if colors else "#4477AA"  # sensible colorblind-safe blue default
	cmap = _make_sequential_from_color(base, PROJECT_HEATMAP_SEQ_NAME)
	try:
		mpl_cmaps.register(cmap, name=PROJECT_HEATMAP_SEQ_NAME)
	except Exception:
		# Already registered or backend does not support re-registration
		pass
	plt.rcParams["image.cmap"] = PROJECT_HEATMAP_SEQ_NAME

def use_default_style() -> None:
	"""Apply the global default style for all subsequent plots."""
	plt.style.use(DEFAULT_STYLES)
	_apply_project_cmaps()
	plt.rcParams["savefig.transparent"] = False
	plt.rcParams["figure.facecolor"] = "white"
	plt.rcParams["axes.facecolor"] = "white"


def use_paper_style() -> None:
	"""Apply the style intended for final paper figures."""
	plt.style.use(PAPER_STYLES)
	_apply_project_cmaps()

	# IF YOU DONOT HAVE LATEX INSTALLED, YOU CAN COMMENT THESE OUT
	plt.rcParams["text.usetex"] = True
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Computer Modern Roman", "DejaVu Serif"]
	plt.rcParams["axes.titlepad"] = 6
	plt.rcParams["axes.labelpad"] = 4


def use_exploratory_style() -> None:
	"""Apply the style intended for exploratory / notebook work."""
	plt.style.use(EXPLORATORY_STYLES)
	_apply_project_cmaps()


def new_figure(
	kind: Literal["single", "double", "custom"] = "single",
	figsize: Optional[Tuple[float, float]] = None,
	nrows: int = 1,
	ncols: int = 1,
	**kwargs,
):
	"""
	Create a new figure and axes with the project default sizes.

	kind:
	    - "single": single-column figure size
	    - "double": double-column figure size
	    - "custom": use `figsize` argument directly
	nrows, ncols:
	    Passed to plt.subplots.
	kwargs:
	    Forwarded to plt.subplots.

	Returns
	-------
	fig, axes
	"""
	if kind == "single":
		_figsize = SINGLE_COL_SIZE
	elif kind == "double":
		_figsize = DOUBLE_COL_SIZE
	else:  # "custom"
		_figsize = figsize if figsize is not None else SINGLE_COL_SIZE
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=_figsize, **kwargs)
	return fig, axes


# ---------------------------
# Generic axis formatters
# ---------------------------
def setup_axes(
	ax,
	xlabel: Optional[str] = None,
	ylabel: Optional[str] = None,
	title: Optional[str] = None,
	grid: bool = True,
	grid_axis: str = "both",
):
	"""Apply common formatting shared across most 2D Cartesian plots."""
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	if grid:
		ax.grid(True, axis=grid_axis)


def setup_scatter_axes(
	ax,
	xlabel: Optional[str] = None,
	ylabel: Optional[str] = None,
	title: Optional[str] = None,
):
	"""Formatting for scatter plots (e.g., topology summaries vs. performance)."""
	setup_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=True, grid_axis="both")


def setup_bar_axes(
	ax,
	xlabel: Optional[str] = None,
	ylabel: Optional[str] = None,
	title: Optional[str] = None,
	rotate_xticks: bool = False,
	rotation: float = 30.0,
):
	"""Formatting for bar plots (e.g., method comparisons)."""
	setup_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=True, grid_axis="y")
	if rotate_xticks:
		for label in ax.get_xticklabels():
			label.set_rotation(rotation)
			label.set_ha("right")


def setup_line_axes(
	ax,
	xlabel: Optional[str] = None,
	ylabel: Optional[str] = None,
	title: Optional[str] = None,
):
	"""Formatting for line plots (e.g., training curves, hyperparameter sweeps)."""
	setup_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=True, grid_axis="both")


def setup_heatmap_axes(
	ax,
	xlabel: Optional[str] = None,
	ylabel: Optional[str] = None,
	title: Optional[str] = None,
	rotate_xticks: bool = False,
	rotation: float = 45.0,
):
	"""
	Formatting for heatmaps (e.g. distance matrices, confusion matrices, persistence summaries).
	Should work for both Matplotlib imshow/pcolormesh and seaborn.heatmap.
	"""
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	if rotate_xticks:
		for label in ax.get_xticklabels():
			label.set_rotation(rotation)
			label.set_ha("right")
	# Typically no additional grid; ticklines and colormap structure are enough.


def setup_violin_axes(
	ax,
	xlabel: Optional[str] = None,
	ylabel: Optional[str] = None,
	title: Optional[str] = None,
	rotate_xticks: bool = False,
	rotation: float = 30.0,
):
	"""
	Formatting for violin plots (e.g. distributions of accuracy, topological
	invariants across runs, etc.).
	Works with both seaborn.violinplot and matplotlib.axes.Axes.violinplot.
	"""
	# For violins, we typically want a y-grid only (across categories).
	setup_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=True, grid_axis="y")
	if rotate_xticks:
		for label in ax.get_xticklabels():
			label.set_rotation(rotation)
			label.set_ha("right")


# ---------------------------
# Radar (polar) helpers
# ---------------------------
def new_radar_figure(
	labels: Iterable[str],
	kind: Literal["single", "double", "custom"] = "single",
	figsize: Optional[Tuple[float, float]] = None,
):
	"""
	Create a radar plot figure and axis with consistent styling.

	Parameters
	----------
	labels : iterable of str
	    Names of the dimensions displayed around the radar.
	kind, figsize : same semantics as new_figure().

	Returns
	-------
	fig, ax, angles
	    fig : Figure
	    ax : PolarAxesSubplot
	    angles : np.ndarray of angles for each label (including closing angle).
	"""
	labels = list(labels)
	L = max(1, len(labels))
	if kind == "single":
		_size = SINGLE_COL_SIZE
	elif kind == "double":
		_size = DOUBLE_COL_SIZE
	else:
		_size = figsize if figsize is not None else SINGLE_COL_SIZE
	# Polar subplot
	fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=_size)
	angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
	angles = np.concatenate([angles, angles[:1]])  # close the loop
	return fig, ax, angles


def setup_radar_axes(
	ax,
	labels: Iterable[str],
	title: Optional[str] = None,
):
	"""
	Apply formatting for a radar axis:
	- Tick labels to match `labels`
	- Optional title
	- Subtle radial grid lines
	"""
	lbls = list(labels)
	L = max(1, len(lbls))
	thetas = np.linspace(0, 2 * np.pi, L, endpoint=False)
	ax.set_xticks(thetas)
	ax.set_xticklabels(lbls)
	ax.set_theta_offset(np.pi / 2.0)
	ax.set_theta_direction(-1)
	# A bit of breathing room on radial labels
	ax.set_rlabel_position(0.0)
	ax.grid(True, alpha=0.3)
	if title:
		ax.set_title(title, y=1.08)


def get_heatmap_cmap(kind: Literal["sequential", "diverging"] = "sequential"):
	"""
	Return a Matplotlib colormap for heatmaps consistent with project style.
	"""
	if kind == "diverging":
		return plt.get_cmap(HEATMAP_CMAP_DIVERGING)
	# For sequential, use the registered project-specific map derived from color cycle
	try:
		return plt.get_cmap(PROJECT_HEATMAP_SEQ_NAME)
	except ValueError:
		# Fallback to static sequential if registration is missing
		return plt.get_cmap(HEATMAP_CMAP_SEQUENTIAL)


