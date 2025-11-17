import logging
import sys
import os
import signal
import matplotlib.pyplot as plt
import matplotlib
# Set Agg backend early to avoid GUI issues
try:
    matplotlib.use('Agg', force=False)
except Exception:
    pass
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - import required for 3D projection
from ripser import ripser
from typing import Dict, Optional
from contextlib import contextmanager
from .plot_style import new_figure

# Patch matplotlib's Path.__deepcopy__ to prevent infinite recursion
# This is a workaround for matplotlib bug where Path.__deepcopy__ calls
# copy.deepcopy(super()) which creates infinite recursion due to circular references
try:
    import matplotlib.path as mpath
    import copy
    
    # Store original method
    if hasattr(mpath.Path, '__deepcopy__'):
        _original_path_deepcopy = mpath.Path.__deepcopy__
    else:
        _original_path_deepcopy = None
    
    def _patched_path_deepcopy(self, memo):
        """
        Patched deepcopy that prevents infinite recursion in Path objects.
        The bug is in matplotlib's Path.__deepcopy__ which calls copy.deepcopy(super())
        causing infinite recursion. We fix this by avoiding the super() deepcopy entirely.
        """
        # Check memo first (standard deepcopy pattern to prevent cycles)
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        
        # Limit recursion by memo size - if we're too deep, use shallow copy
        if len(memo) > 100:
            # We're too deep - break the cycle with shallow copy
            result = copy.copy(self)
            memo[id_self] = result
            return result
        
        # Create new Path with copied vertices and codes, avoiding super() call
        # This is the key fix - we don't call super().__deepcopy__ which causes the recursion
        try:
            # Copy vertices (numpy array)
            if hasattr(self.vertices, 'copy'):
                vertices = self.vertices.copy()
            else:
                vertices = self.vertices
            
            # Copy codes (can be None or array)
            if self.codes is not None and hasattr(self.codes, 'copy'):
                codes = self.codes.copy()
            else:
                codes = self.codes
            
            # Create new Path object
            new_path = mpath.Path(
                vertices,
                codes,
                _interpolation_steps=getattr(self, '_interpolation_steps', 1)
            )
            
            # Store in memo before returning
            memo[id_self] = new_path
            return new_path
        except (RecursionError, Exception) as e:
            # If anything fails, fall back to shallow copy
            result = copy.copy(self)
            memo[id_self] = result
            return result
    
    # Apply the patch
    if _original_path_deepcopy is not None:
        mpath.Path.__deepcopy__ = _patched_path_deepcopy
        # Verify patch was applied by checking if it's our function
        if mpath.Path.__deepcopy__ is _patched_path_deepcopy:
            logging.info("Successfully applied matplotlib Path.__deepcopy__ patch to prevent recursion")
            # Test the patch works by trying to deepcopy a simple path
            try:
                import copy as copy_module
                test_path = mpath.Path([[0, 0], [1, 1]])
                test_copy = copy_module.deepcopy(test_path)
                logging.debug("Path deepcopy patch verified - test deepcopy succeeded")
            except RecursionError:
                logging.warning("Path deepcopy patch may not be working - test deepcopy failed with RecursionError")
            except Exception as e:
                logging.debug(f"Path deepcopy patch test: {type(e).__name__} (may be expected)")
        else:
            logging.warning("Failed to apply Path.__deepcopy__ patch - method was not replaced")
    else:
        logging.debug("matplotlib.Path.__deepcopy__ not found, skipping patch")
except (ImportError, AttributeError, Exception) as e:
    # If patching fails, continue without it (non-critical)
    logging.warning(f"Could not patch matplotlib Path.__deepcopy__: {type(e).__name__}: {e}")
    pass


@contextmanager
def increased_recursion_limit(limit=5000):
    """
    Temporarily increase recursion limit to handle matplotlib deepcopy issues with 3D plots.
    This is a workaround for matplotlib's Path.__deepcopy__ recursion bug.
    """
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(old_limit)


def _save_figure_safe(fig, save_path: str, has_3d: bool = False) -> bool:
    """
    Safely save a matplotlib figure, handling recursion errors for 3D plots.
    For 3D plots, the recursion happens in matplotlib's Path.__deepcopy__ method
    which has a circular reference bug. We use multiple strategies to work around this.
    Returns True if successful, False otherwise.
    """
    if not has_3d:
        # For 2D plots, use tight bbox normally
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            return True
        except RecursionError:
            # Fallback without tight bbox
            fig.savefig(save_path, dpi=150)
            return True
    
    # For 3D plots: The recursion is in Path.__deepcopy__ which matplotlib calls
    # during save operations. The issue is that matplotlib's Path objects have
    # circular references in their deepcopy implementation.
    
    # Strategy 1: Try with the patched deepcopy (should work now)
    try:
        fig.savefig(save_path, dpi=150, format='png')
        return True
    except RecursionError:
        pass
    except Exception as e:
        logging.debug(f"Strategy 1 failed: {type(e).__name__}")
        pass
    
    # Strategy 2: Try with increased recursion limit (in case patch didn't fully work)
    try:
        with increased_recursion_limit(10000):
            fig.savefig(save_path, dpi=150, format='png')
        return True
    except RecursionError:
        pass
    except Exception as e:
        logging.debug(f"Strategy 2 failed: {type(e).__name__}")
        pass
    
    # Strategy 3: Try direct canvas print_png (lowest level, might bypass deepcopy)
    try:
        if hasattr(fig.canvas, 'print_png'):
            with increased_recursion_limit(10000):
                # Set DPI on figure before calling print_png (print_png doesn't accept dpi argument)
                original_dpi = fig.dpi
                fig.set_dpi(150)
                fig.canvas.print_png(save_path)
                fig.set_dpi(original_dpi)  # Restore original DPI
            return True
    except (RecursionError, AttributeError, RuntimeError, TypeError) as e:
        logging.debug(f"Strategy 3 failed: {type(e).__name__}")
        pass
    
    # All strategies failed - this is a known matplotlib bug we cannot fully fix
    # The Path.__deepcopy__ has a circular reference that causes infinite recursion
    # Even with the patch and workarounds, matplotlib's internal save operations may still trigger it
    logging.warning("All save strategies failed for 3D plot - matplotlib Path deepcopy bug persists")
    logging.warning("This is a known matplotlib limitation - visualization will be skipped (non-critical)")
    return False


def visualize_sample_diagrams(
    point_clouds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    n_samples_per_class: int = 1,
    maxdim: int = 2,
    save_path: Optional[str] = "persistence_diagrams_by_class.png",
    show: bool = False,
    seed: Optional[int] = 42,
):
    """
    Visualize point clouds and persistence diagrams for a few samples per class.
    - Infers classes from 'labels' unless 'class_names' provided as mapping {class_id: name}.
    - Supports arbitrary number of classes and maxdim (0..2+).
    - For point clouds with D==2 uses 2D scatter; D>=3 uses first 3 dims in 3D.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    unique_classes = np.unique(labels)
    unique_classes = list(sorted(unique_classes.tolist()))
    num_classes = len(unique_classes)
    if num_classes == 0:
        return None

    # layout: rows = num_classes * n_samples_per_class, cols = 1 (cloud) + maxdim+1 (diagrams)
    rows = num_classes * max(1, n_samples_per_class)
    cols = 1 + (maxdim + 1)
    fig, _ = new_figure(kind="custom", figsize=(4 * cols, 3 * rows))
    
    # Track if we have any 3D plots (which can cause recursion issues with bbox_inches='tight')
    has_3d_plots = False

    def _name_for(c):
        key = None
        try:
            key = int(c)
        except Exception:
            key = c
        if class_names and key in class_names:
            return class_names[key]
        return f"Class {key}"

    for ci, c in enumerate(unique_classes):
        class_mask = labels == c
        class_pcs = point_clouds[class_mask]
        if len(class_pcs) == 0:
            continue
        k = min(len(class_pcs), max(1, n_samples_per_class))
        # sample without replacement
        sample_idx = rng.choice(len(class_pcs), size=k, replace=False)
        for sj, idx_sel in enumerate(sample_idx):
            row = ci * max(1, n_samples_per_class) + sj
            pc = class_pcs[idx_sel]
            # diagrams
            try:
                diagrams = ripser(pc, maxdim=maxdim)['dgms']
            except Exception:
                diagrams = []

            # point cloud subplot
            D = pc.shape[1] if pc.ndim == 2 else 3
            if D >= 3:
                ax = fig.add_subplot(rows, cols, row * cols + 1, projection='3d')
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, alpha=0.7)
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
                ax.view_init(elev=20, azim=45)
                has_3d_plots = True
            elif D == 2:
                ax = fig.add_subplot(rows, cols, row * cols + 1)
                ax.scatter(pc[:, 0], pc[:, 1], s=4, alpha=0.8)
                ax.set_xlabel('X'); ax.set_ylabel('Y')
            else:
                ax = fig.add_subplot(rows, cols, row * cols + 1)
                ax.text(0.5, 0.5, f"{_name_for(c)}\nD={D} not visualized", ha="center", va="center")
                ax.axis('off')
            if sj == 0:
                ax.set_title(f'{_name_for(c)} - Point Cloud', fontsize=10)

            # diagram subplots for H0..Hmax
            for kdim in range(0, maxdim + 1):
                ax = fig.add_subplot(rows, cols, row * cols + 2 + kdim)
                if kdim < len(diagrams) and len(diagrams[kdim]) > 0:
                    dgm = diagrams[kdim]
                    finite = dgm[np.isfinite(dgm[:, 1])]
                    if len(finite) > 0:
                        ax.scatter(finite[:, 0], finite[:, 1], s=10, alpha=0.7)
                        mx = max(float(finite[:, 0].max()), float(finite[:, 1].max()))
                        ax.plot([0, mx], [0, mx], 'k--', linewidth=1)
                ax.set_xlabel('Birth'); ax.set_ylabel('Death')
                if sj == 0:
                    ax.set_title(f'H{kdim}', fontsize=10)
                ax.grid(True, alpha=0.3)

    # tight_layout can be problematic with 3D plots, so wrap in try-except
    try:
        plt.tight_layout()
    except (RecursionError, Exception):
        # If tight_layout fails (e.g., with 3D plots), continue without it
        pass
    
    if save_path:
        # Use safe save function that handles recursion errors
        # Wrap in additional try-except to catch any recursion errors that slip through
        try:
            success = _save_figure_safe(fig, save_path, has_3d=has_3d_plots)
            if success:
                print(f"Saved persistence diagrams to '{save_path}'")
            else:
                print(f"Could not save persistence diagrams")
                logging.warning(f"Failed to save persistence_diagrams_by_class.png")
        except RecursionError as e:
            # Catch any recursion errors that weren't caught by _save_figure_safe
            # This is critical - ensures pipeline continues even if recursion occurs
            logging.warning(f"RecursionError caught during figure save - skipping visualization (non-critical): {type(e).__name__}")
            print(f"Could not save persistence diagrams (matplotlib recursion bug - non-critical, pipeline continues)")
        except Exception as e:
            # Catch any other unexpected errors
            logging.warning(f"Error during figure save (non-critical, pipeline continues): {type(e).__name__}: {e}")
            print(f"Could not save persistence diagrams (non-critical)")
    
    # Always close the figure to free memory, even if save failed
    try:
        if show:
            plt.show()
        else:
            plt.close(fig)
    except Exception:
        # Even closing might fail in extreme cases, but we continue
        pass
    
    return fig


def plot_original_vs_adversarial(x_orig, x_adv, title_suffix: str = ""):
    """
    Show original vs adversarial point cloud more clearly:

    - left: overlay original/adversarial with same axes
    - right: displacement lines from original -> adversarial
    """
    if isinstance(x_orig, torch.Tensor):
        x_orig = x_orig.detach().cpu().numpy()
    if isinstance(x_adv, torch.Tensor):
        x_adv = x_adv.detach().cpu().numpy()
    x_adv = x_adv.squeeze(0)

    # Common axis limits so scaling doesn't hide differences
    all_pts = np.vstack([x_orig, x_adv])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    fig, _ = new_figure(kind="custom", figsize=(12, 4))

    # ---- Left: overlay original + adversarial ----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2],
                s=8, alpha=0.8, label="original")
    ax1.scatter(x_adv[:, 0], x_adv[:, 1], x_adv[:, 2],
                s=8, alpha=0.8, marker="^", label="adversarial")
    ax1.set_title(f"Overlay {title_suffix}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.legend()

    # ---- Right: displacement lines ----
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2],
                s=5, alpha=0.5, color="gray")
    ax2.scatter(x_adv[:, 0], x_adv[:, 1], x_adv[:, 2],
                s=8, alpha=0.9, color="tab:orange")

    # connect each original point to its adversarial point
    for i in range(x_orig.shape[0]):
        xs = [x_orig[i, 0], x_adv[i, 0]]
        ys = [x_orig[i, 1], x_adv[i, 1]]
        zs = [x_orig[i, 2], x_adv[i, 2]]
        ax2.plot(xs, ys, zs, linewidth=0.5, color="tab:orange", alpha=0.7)

    ax2.set_title(f"Displacement vectors {title_suffix}")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.show()


def plot_torus_wireframe_compare(x_orig, x_adv, title_suffix: str = "(Torus)"):
    """
    Side-by-side wireframe torus: original vs adversarial, with colors.

    Assumes point clouds from make_point_clouds torus:
    N = n_points**2, ordered as nested loops.
    """
    # to numpy
    if isinstance(x_orig, torch.Tensor):
        x_orig = x_orig.detach().cpu().numpy()
    if isinstance(x_adv, torch.Tensor):
        x_adv = x_adv.detach().cpu().numpy()
    x_adv = x_adv.squeeze(0)  # (N, 3) from (1, N, 3)

    N = x_orig.shape[0]
    n = int(round(np.sqrt(N)))
    if n * n != N:
        print(f"[wireframe] N={N} is not a perfect square; falling back to scatter.")
        return

    orig_grid = x_orig.reshape(n, n, 3)
    adv_grid  = x_adv.reshape(n, n, 3)

    # common axis limits so shapes are directly comparable
    all_pts = np.vstack([x_orig, x_adv])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    fig, _ = new_figure(kind="custom", figsize=(12, 5))

    # --- Original wireframe ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # color lines by parameter index to give structure
    colors_u = plt.cm.Blues(np.linspace(0.3, 0.9, n))
    colors_v = plt.cm.Purples(np.linspace(0.3, 0.9, n))

    for i in range(n):  # vary first parameter
        ax1.plot(orig_grid[i, :, 0], orig_grid[i, :, 1], orig_grid[i, :, 2],
                 linewidth=1.0, color=colors_u[i], alpha=0.9)
    for j in range(n):  # vary second parameter
        ax1.plot(orig_grid[:, j, 0], orig_grid[:, j, 1], orig_grid[:, j, 2],
                 linewidth=0.8, color=colors_v[j], alpha=0.7)

    ax1.set_title(f"Original {title_suffix}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax1.set_box_aspect((1, 1, 1))

    # --- Adversarial wireframe ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    colors_u2 = plt.cm.Oranges(np.linspace(0.3, 0.9, n))
    colors_v2 = plt.cm.Greens(np.linspace(0.3, 0.9, n))

    for i in range(n):
        ax2.plot(adv_grid[i, :, 0], adv_grid[i, :, 1], adv_grid[i, :, 2],
                 linewidth=1.0, color=colors_u2[i], alpha=0.9)
    for j in range(n):
        ax2.plot(adv_grid[:, j, 0], adv_grid[:, j, 1], adv_grid[:, j, 2],
                 linewidth=0.8, color=colors_v2[j], alpha=0.7)

    ax2.set_title(f"Adversarial {title_suffix}")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
    ax2.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.show()


