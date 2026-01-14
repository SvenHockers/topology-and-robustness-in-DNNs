# Narrowed hypothesis (topology × robustness)

## Hypothesis (narrow, testable)
Let \(z=\phi_\ell(x)\) be a fixed intermediate representation of a trained classifier and let \(N_k(x)\) be a kNN neighborhood of \(z\) in the training set. Consider persistent-homology (PH) features computed on the local point cloud \(\{z\}\cup N_k(x)\).

**H1 (conditional):** *When* (i) neighborhoods are chosen in a **class-conditional** way (e.g., restricted to the predicted class), and (ii) local neighborhoods are made **metric-comparable** across the manifold (e.g., via local centering/whitening), then **adversarial/OOD samples are enriched among points whose metric-conditioned PH features are atypical for the predicted class** relative to clean validation data.

## What this does *not* claim
- It does **not** claim that adversarial points must be **class-mixed** under *global* kNN (purity may remain high).
- It does **not** claim universality across modalities/datasets; it predicts failure in regimes where representation neighborhoods are not topology-informative (often tabular MLP embeddings).

## Observable predictions (what the notebook should show)
1. **Ablation:** global vs class-conditional neighborhood selection changes detector reliability when clean features are multi-modal across classes.
2. **Ablation:** local metric conditioning (whiten / scale) improves separation if raw distances are dominated by anisotropy/density.
3. **Mechanistic stratification:** the effect strengthens on **successful** attacks and/or low-margin points (decision boundary proximity), but the detector must be evaluated on **all** attacks for practical operating points.

