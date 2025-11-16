import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - import required for 3D projection
from ripser import ripser


def visualize_sample_diagrams(point_clouds: np.ndarray, labels: np.ndarray):
    """
    Visualize point cloud and persistence diagrams (H0, H1, H2) for one sample from each class.
    """
    class_names = ['Circle', 'Sphere', 'Torus']

    fig = plt.figure(figsize=(16, 12))

    for class_idx in range(3):
        # Get one sample from this class
        class_mask = labels == class_idx
        class_pcs = point_clouds[class_mask]
        pc = class_pcs[0]  # Take first sample

        # Compute persistence diagrams
        diagrams = ripser(pc, maxdim=2)['dgms']

        # Plot 3D point cloud
        ax = fig.add_subplot(3, 4, class_idx * 4 + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='blue', s=1, alpha=0.6)
        ax.set_title(f'{class_names[class_idx]} - Point Cloud', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)

        # Plot H0 diagram
        ax = fig.add_subplot(3, 4, class_idx * 4 + 2)
        if len(diagrams[0]) > 0:
            finite_dgm = diagrams[0][diagrams[0][:, 1] != np.inf]
            if len(finite_dgm) > 0:
                ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], c='blue', s=20, alpha=0.6)
            # Plot diagonal
            max_val = max(diagrams[0][:, 0].max(), finite_dgm[:, 1].max() if len(finite_dgm) > 0 else 1)
            ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title('H₀ (Components)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot H1 diagram
        ax = fig.add_subplot(3, 4, class_idx * 4 + 3)
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            finite_dgm = diagrams[1][diagrams[1][:, 1] != np.inf]
            if len(finite_dgm) > 0:
                ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], c='orange', s=20, alpha=0.6)
                max_val = max(finite_dgm[:, 0].max(), finite_dgm[:, 1].max())
                ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title('H₁ (Loops)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot H2 diagram
        ax = fig.add_subplot(3, 4, class_idx * 4 + 4)
        if len(diagrams) > 2 and len(diagrams[2]) > 0:
            finite_dgm = diagrams[2][diagrams[2][:, 1] != np.inf]
            if len(finite_dgm) > 0:
                ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], c='green', s=20, alpha=0.6)
                max_val = max(finite_dgm[:, 0].max(), finite_dgm[:, 1].max())
                ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title('H₂ (Voids)', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('persistence_diagrams_by_class.png', dpi=150, bbox_inches='tight')
    print("Saved persistence diagrams to 'persistence_diagrams_by_class.png'")
    plt.show()


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


