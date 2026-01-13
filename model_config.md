### Frozen model configurations (for detector sweeps)

We freeze a **single classifier per dataset** so that detector sweeps are comparable. The key requirement is: the classifier must be **respectable on clean data** while still being **meaningfully foolable** by our chosen adversarial budget, otherwise detector metrics collapse (if attacks don’t succeed) or become trivial (if attacks are always successful).

All selections below were guided by the empirical outputs in:
- `results/model_foolability_breast_cancer_tabular.csv`
- `results/model_foolability_geometrical-shapes.csv`
- `results/model_foolability_mnist.csv` and `results/model_foolability_mnist_sanity.csv`

#### Summary table

| Dataset (registry key) | Input modality | Frozen model | Model kwargs | Training config (frozen) | Why this model was selected |
|---|---|---|---|---|---|
| `breast_cancer_tabular` | tabular vector (standardized) | `two_moons_mlp` | `input_dim` inferred (=30), `output_dim=2` | hidden `[128, 64]`, activation `relu`, lr `1e-3`, weight_decay `1e-4`, epochs `80`, batch `64` | Best “good + attackable” tradeoff in the quick search: **~0.96 test accuracy** while PGD attack success at eps `0.2` is **~0.21**. Smaller models lost accuracy without becoming meaningfully more foolable. |
| `geometrical-shapes` | 3D point cloud (vector) | `two_moons_mlp` | `input_dim` inferred (=3), `output_dim=2` | hidden `[256, 128]`, activation `relu`, lr `1e-3`, weight_decay `0.0`, epochs `40`, batch `64` | Achieves **near-perfect clean accuracy** while being the **most foolable** among tested MLPs (PGD success **~0.18 at eps 0.5** vs ~0.13–0.14 for smaller variants). This dataset can still show low success at small eps; final detector experiments may require a stronger attack budget (higher eps / more steps). |
| `mnist` | image (torchvision, `[0,1]`) | `minicnn` | `in_channels=1`, `feat_dim=128`, `num_classes=10` | lr `1e-3`, weight_decay `1e-4`, epochs `20`, batch `64` | Sanity-checked to provide a **non-saturated foolability regime**: with PGD, success is **~0.23 at eps 0.05**, **~0.69 at eps 0.1**, **~0.93 at eps 0.2**, while clean accuracy remains **~0.95** on the evaluated split. This gives both “moderate” and “strong” attack regimes for detector evaluation. |

#### Notes / constraints (important for interpretation)

- **Detector layer**: for feature-space scoring we only use the **penultimate layer** (fixed).
- **Why we care about attack success rate**: detector evaluation on adversarial data is only meaningful if a non-trivial fraction of points are successfully attacked. If success is ~0, adversarial ≈ clean and AUROC tends to ~0.5.
- **Robustness of the choice**: the foolability CSVs are based on specific subsampling / seeds. Before publishing final results, it’s good practice to re-check each frozen model across **multiple seeds** (even 3) to ensure the clean accuracy and attack success don’t swing heavily.

