
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to path
notebook_dir = "/Users/svenhockers/Desktop/Graph Manifold/notebooks"
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import generate_two_moons
from src.graph_manifold import compute_laplacian_smoothness_score
from src.utils import set_seed

# Set seed for reproducibility
SEED = 42
set_seed(SEED)

print("Path setup complete.")

# Generate clean Two Moons dataset
n_samples = 1000
noise = 0.1
X_train, y_train_clean, _, _, _, _ = generate_two_moons(
    n_samples=n_samples,
    noise=noise,
    random_state=SEED
)

print(f"Generated {len(X_train)} training samples")

# Introduce label noise (flip labels)
flip_ratio = 0.10  # 10% label noise
n_flip = int(len(y_train_clean) * flip_ratio)

# Choose random indices to flip
np.random.seed(SEED)  
flip_indices = np.random.choice(len(y_train_clean), n_flip, replace=False)

# Create noisy label vector
y_train_noisy = y_train_clean.copy()
y_train_noisy[flip_indices] = 1 - y_train_noisy[flip_indices]  # Flip 0->1, 1->0

# Create ground truth mask for flipped labels (1 = flipped, 0 = clean)
is_flipped = np.zeros(len(y_train_clean), dtype=int)
is_flipped[flip_indices] = 1

print(f"Flipped {n_flip} labels ({flip_ratio*100}%)")

# Compute Laplacian scores
k_neighbors = 10
scores = np.zeros(len(X_train))

print("Computing scores...")
for i in range(len(X_train)):
    z = X_train[i]
    f_z = y_train_noisy[i]
    
    scores[i] = compute_laplacian_smoothness_score(
        z=z,
        f_z=f_z,
        Z_train=X_train,
        f_train=y_train_noisy,
        k=k_neighbors
    )

print(f"Scores computed. Range: [{scores.min():.4f}, {scores.max():.4f}]")

# Quantitative Evaluation
auc = roc_auc_score(is_flipped, scores)
print(f"Detection AUC: {auc:.4f}")

# Check simple stats
clean_mean = scores[is_flipped==0].mean()
flipped_mean = scores[is_flipped==1].mean()
print(f"Mean score (Clean): {clean_mean:.4f}")
print(f"Mean score (Flipped): {flipped_mean:.4f}")

if auc > 0.8:
    print("SUCCESS: High AUC achieved.")
else:
    print("WARNING: AUC might be too low.")
