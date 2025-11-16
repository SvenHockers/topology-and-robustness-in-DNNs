import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - import required for 3D projection
from ripser import ripser
from typing import Dict, Optional


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
    fig = plt.figure(figsize=(4 * cols, 3 * rows))

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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved persistence diagrams to '{save_path}'")
    if show:
        plt.show()
    else:
        plt.close(fig)
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

    fig = plt.figure(figsize=(12, 4))

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

    fig = plt.figure(figsize=(12, 5))

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


