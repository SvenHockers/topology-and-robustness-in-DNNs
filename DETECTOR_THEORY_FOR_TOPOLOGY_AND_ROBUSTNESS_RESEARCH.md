# Graph-Manifold Off-Manifold Detection (Researcher Notes)

This document explains the detector implemented in `src/detectors.py` and used in `notebooks/01_graph_manifold_two_moons.ipynb`, with an emphasis on **mathematical foundations** relevant to ML research on **geometry/topology** and **model robustness**.

The implementation is intentionally modular:

- **Score construction** (geometry / spectral proxies): `src/graph_manifold.py` (`compute_graph_scores`, `compute_degree_score`, `compute_laplacian_smoothness_score`, optional diffusion maps, and local tangent residuals).
- **Detector** (decision rule on 1D scores or small feature vector): `src/detectors.py` (`ScoreBasedDetector`, `SupervisedGraphDetector`).
- **Notebook pipeline**: `notebooks/01_graph_manifold_two_moons.ipynb` builds a reference set in **feature space**, computes scores on clean/adversarial points, calibrates a threshold on clean validation scores, and evaluates.

---

## 1) Setup and notation

### Data + representation manifold

Let \(x \in \mathcal{X}\) be an input, and let \(z=\phi(x)\in\mathbb{R}^d\) be a representation (either input space or a learned feature space). The notebook typically uses **penultimate-layer** features:

- `config.graph.space = 'feature'`
- \(Z_{\text{train}} = \{\phi(x_i)\}_{i=1}^n\)

Assumption (manifold model, standard in manifold learning):

- There exists a compact, smooth \(m\)-dimensional submanifold \(M \subset \mathbb{R}^d\) with \(m \ll d\),
- Training representations \(\{z_i\}\) are sampled near \(M\) (possibly with noise).

### Graph construction (kNN + Gaussian weights)

Given the reference set \(Z_{\text{train}}=\{z_i\}_{i=1}^n\), the code constructs a weighted, undirected kNN graph with weights

\[
w_{ij} = \exp\!\left(-\frac{\|z_i-z_j\|^2}{2\sigma^2}\right)
\quad \text{if } z_j \in \text{kNN}(z_i),
\]

then symmetrizes \(W=(W+W^\top)/2\). Degrees are \(d_i=\sum_j w_{ij}\). (See `build_knn_graph`.)

The scale \(\sigma\) defaults to a **median distance heuristic** in the code.

### Model outputs used by smoothness scores

For each point \(x\), the notebook uses the model’s class-1 probability

\[
f(x)=\Pr_\theta(y=1\mid x),
\]

computed as softmax over logits (see `compute_graph_scores` and notebook cell where `f_train = probs_train[:, 1]`).

---

## 2) Scores: what they measure and why they work

`compute_graph_scores` returns a dictionary of scalar score arrays. In the two-moons notebook, the computed keys are:

- `degree`
- `laplacian`
- `tangent_residual`
- `tangent_residual_z`
- `knn_radius`
- (optionally) `diffusion`
- (optionally) `combined` (computed in `src/detectors.py` when requested)

These are **one-dimensional test statistics** intended to be large when a point is “off-manifold” or “inconsistent with manifold-smooth predictions.”

---

## 2.1 Degree score: kernel density proxy

### Implementation

For a query point \(z\), the code finds its k nearest neighbors in \(Z_{\text{train}}\), computes Gaussian weights to them, and defines the **degree**

\[
\deg(z) \;=\; \sum_{j\in \text{kNN}(z)} \exp\!\left(-\frac{\|z-z_j\|^2}{2\sigma^2}\right).
\]

It returns the score as

\[
s_{\text{deg}}(z) = -\deg(z),
\]

so **larger score means more suspicious** (lower connectivity / lower density).

### Mathematical foundation

For kernel graphs, \(\deg(z)\) is a close relative of a kernel density estimator (KDE). Under i.i.d. sampling from a smooth density \(p\) on \(M\), for bandwidth \(\sigma\to 0\) with appropriate scaling,

\[
\deg(z) \approx C(\sigma,m)\, n\, p(z)
\]

up to curvature- and boundary-dependent corrections. Thus, \(-\deg(z)\) behaves like an **inverse density** statistic: points in low-density regions of the representation manifold score high.

Research intuition: many adversarial examples in feature space move into **locally low-density** regions relative to the training manifold, especially for “off-manifold” perturbations.

---

## 2.2 Laplacian smoothness score: local Dirichlet energy increment

### Implementation

For a query point \(z\) and its model output \(f(z)\), the code computes

\[
s_{\text{lap}}(z) \;=\; \sum_{j\in \text{kNN}(z)} w(z,z_j)\,\bigl(f(z)-f(z_j)\bigr)^2.
\]

This is exactly what `compute_laplacian_smoothness_score` returns as `delta_E`.

### Relation to the graph Laplacian quadratic form

For a graph signal \(f\in\mathbb{R}^n\), the (unnormalized) graph Dirichlet energy is

\[
\mathcal{E}_G(f) = \frac12 \sum_{i,j} w_{ij} (f_i-f_j)^2 = f^\top L f,
\quad L=D-W.
\]

If you “attach” a new node \(z\) to the graph with edges \(w(z,z_j)\), then the *increment* in energy contributed by those new edges is exactly

\[
\Delta \mathcal{E}(z) = \sum_{j} w(z,z_j)\,(f(z)-f_j)^2,
\]

which is precisely the score in the code (up to constant factors depending on conventions).

### Manifold-learning theorem connection

One of the central results in spectral manifold learning is that (properly normalized) graph Laplacians converge to differential operators on the underlying manifold:

- **Belkin–Niyogi type results**: as \(n\to\infty\) and \(\sigma\to 0\) with correct scaling, graph Laplacians (and graph Dirichlet energies) converge to the **Laplace–Beltrami operator** \(\Delta_M\) (or density-weighted variants) on \(M\).

In that asymptotic regime, the graph Dirichlet energy approximates a continuum energy of the form

\[
\int_M \|\nabla_M f\|^2 \,\rho \, d\mathrm{vol}_M
\]

for a density-related weight \(\rho\). Therefore, a large \(s_{\text{lap}}(z)\) indicates that \(f\) is **not locally smooth** along the manifold neighborhood near \(z\), which is consistent with adversarial behavior that induces sharp local changes in the model’s decision function relative to the training manifold geometry.

---

## 2.3 Local tangent residual: distance to an estimated tangent space

This is the notebook’s primary “off-manifold” score choice (`score_type='tangent_residual_z'` by default in the notebook).

### Implementation (local PCA tangent plane)

For each query \(z\):

1. Collect a neighborhood \(N(z)=\{z_{j_1},\dots,z_{j_k}\}\) (with `tangent_k` neighbors).
2. Center it: \(X_c = [z_{j}-\mu]_{j\in N(z)}\), where \(\mu=\frac1k\sum_{j\in N(z)} z_j\).
3. Fit PCA on \(X_c\). Let \(V\in\mathbb{R}^{r\times d}\) contain the top \(r\) principal directions as **rows** (the code uses `pca.components_[:r]`).
   - If `tangent_dim` is not set, the code chooses the smallest \(r\) such that the cumulative explained variance ratio reaches `tangent_var_threshold` (with bounds `tangent_dim_min`, `tangent_dim_max`).
4. Project the centered query \(z_c=z-\mu\) onto the local PCA subspace:

\[
z_{\text{proj}} = (z_c V^\top)V,
\]

and define the residual

\[
r(z) = z_c - z_{\text{proj}} = \bigl(I - V^\top V\bigr)z_c.
\]

The raw tangent score is the squared residual norm:

\[
s_{\text{tan}}(z) = \|r(z)\|^2.
\]

### Local normalization: `tangent_residual_z`

The code also computes a z-scored version against the neighborhood’s residual distribution:

- For each neighbor \(u \in N(z)\), compute its residual norm squared \(\| (I-V^\top V)(u-\mu) \|^2\).
- Let \(\mu_r\) and \(\sigma_r\) be the mean and standard deviation of those values.
- Define

\[
s_{\text{tan-z}}(z) = \frac{s_{\text{tan}}(z) - \mu_r}{\sigma_r + 10^{-12}}.
\]

This matters when perturbations are small: the *absolute* residual can be tiny, but still **atypical relative to local variation**.

### Mathematical foundation (tangent space estimation)

If data are sampled from a smooth manifold \(M\), then locally (within a sufficiently small neighborhood) \(M\) is well-approximated by its tangent space \(T_{p}M\) at \(p\in M\). Local PCA is a classical estimator of tangent spaces:

- Under standard sampling/noise conditions and neighborhoods shrinking with \(n\), the PCA subspace converges to \(T_{p}M\) (up to rotational ambiguity).

Interpretation:

- For \(z\) near \(M\), \(s_{\text{tan}}(z)\) approximates the squared norm of the **normal component** of \(z-\mu\), i.e., a proxy for squared distance to the local tangent plane, hence a proxy for distance off the manifold (to first order).

---

## 2.4 kNN radius: kNN-density proxy (scale-free)

### Implementation

The code computes mean distance to neighbors:

\[
s_{\text{rad}}(z) \;=\; \frac{1}{k'}\sum_{\ell=2}^{k'+1} \|z-z_{(\ell)}\|,
\]

where \(z_{(\ell)}\) is the \(\ell\)-th nearest neighbor and \(k'=\texttt{graph\_params.k}\) (the code drops the first distance if it is a self-match).

### Mathematical foundation

In \(m\)-dimensional settings, the kNN radius \(r_k(z)\) satisfies the classical kNN density scaling:

\[
r_k(z)^m \approx \frac{k}{n\,p(z)\,v_m},
\]

where \(v_m\) is the volume of the unit \(m\)-ball. Thus \(s_{\text{rad}}(z)\) increases as density decreases.

Compared to \(-\deg(z)\), the kNN radius can be less sensitive to the exact kernel bandwidth \(\sigma\) and easier to interpret as a **local scale** statistic.

---

## 2.5 Diffusion score (optional): diffusion maps + Nyström extension

This path is enabled by `graph_params.use_diffusion = True` (off by default).

### Implementation outline

1. Build a weighted kNN graph on \(Z_{\text{train}}\) with weights \(W\).
2. Apply anisotropic normalization parameterized by \(\alpha\) (code default \(\alpha=0.5\)):

\[
W_\alpha = D^{-\alpha} W D^{-\alpha},
\quad D=\mathrm{diag}(W\mathbf{1}).
\]

3. Row-normalize to a Markov transition matrix \(P\):

\[
P = \tilde{D}^{-1} W_\alpha,\quad \tilde{D}=\mathrm{diag}(W_\alpha \mathbf{1}).
\]

4. Compute leading eigenvectors of \(P^\top\) (code uses `eigs(P.T, ...)`) and use them as the diffusion embedding coordinates (skipping the trivial eigenvalue \(1\)).
5. Embed a new point \(z\) using a Nyström-style weighted average of neighbor embeddings:

\[
\Psi(z) = \sum_{j\in \text{kNN}(z)} \bar{w}_j\,\Psi(z_j),
\quad \bar{w}_j = \frac{w(z,z_j)}{\sum_{\ell} w(z,z_\ell)}.
\]

6. Score the point by its minimal embedding distance to training embeddings:

\[
s_{\text{diff}}(z) = \min_i \|\Psi(z) - \Psi(z_i)\|.
\]

### Mathematical foundation

Diffusion maps (Coifman–Lafon) approximate the heat kernel geometry on \(M\). In the large-sample limit, diffusion distances relate to geodesic structure smoothed by a diffusion time parameter. Practically, diffusion embeddings can be **more robust to noise** and can capture non-linear geometry beyond local PCA.

---

## 2.6 Combined score (optional): standardized linear fusion

If `DetectorConfig.score_type == 'combined'`, `src/detectors.py` constructs

1. Standardize degree and laplacian scores separately (z-score normalization across the provided batch):
   \[
   \tilde{s}_{\text{deg}}=\frac{s_{\text{deg}}-\mu_{\text{deg}}}{\sigma_{\text{deg}}},\quad
   \tilde{s}_{\text{lap}}=\frac{s_{\text{lap}}-\mu_{\text{lap}}}{\sigma_{\text{lap}}}.
   \]
2. Combine:
   \[
   s_{\text{comb}} = \alpha\,\tilde{s}_{\text{deg}} + \beta\,\tilde{s}_{\text{lap}}.
   \]

This is a simple late-fusion baseline; it is *not* a likelihood-ratio optimal combination unless additional assumptions hold.

---

## 3) Decision rules in `src/detectors.py`

There are two detector families.

---

## 3.1 `ScoreBasedDetector`: quantile thresholding (anomaly-style)

### Implementation

Given a chosen score \(s(z)\) (one of the keys above), the detector:

1. Fits a threshold on **clean** validation scores:
   \[
   \tau = \mathrm{Quantile}_{q}( \{ s(z_i): y_i=0\} ),
   \quad q=0.95\ \text{by default}.
   \]
2. Predicts
   \[
   \hat{y}(z) = \mathbf{1}\{ s(z) > \tau \}.
   \]

It also returns a monotone probability proxy:

\[
\hat{p}(z)=\sigma\bigl(s(z)-\tau\bigr)=\frac{1}{1+\exp(-(s(z)-\tau))}.
\]

Important: this \(\hat{p}\) is **not calibrated** as a true posterior; it is just a smooth monotone transform of the score centered at the threshold.

### Statistical guarantee: approximate FPR control via empirical quantiles

Let \(S\) be the random score under the clean distribution with CDF \(F_0\). If \(\tau^\star = F_0^{-1}(q)\), then

\[
\Pr_0(S > \tau^\star) = 1-q.
\]

The detector uses the **empirical** quantile \(\hat{\tau}\) from \(n_0\) clean validation samples. A standard finite-sample tool here is the Dvoretzky–Kiefer–Wolfowitz (DKW) inequality, which bounds \(\sup_s | \hat{F}_0(s)-F_0(s)|\), implying \(\hat{\tau}\) concentrates around \(\tau^\star\). In practice: if the validation clean distribution matches deployment, choosing the 95th percentile targets a ~5% false positive rate (up to sampling error).

This is a key robustness-research point: the procedure controls a **Type-I error rate** against the clean distribution used for calibration, but can drift under representation shift.

---

## 3.2 `SupervisedGraphDetector`: logistic regression on score features

### Implementation

If `DetectorConfig.detector_type == 'supervised'`, the code forms a feature vector

\[
\mathbf{s}(z)=\bigl[s_{\text{deg}}(z),\ s_{\text{lap}}(z),\ (s_{\text{diff}}(z)\text{ if present})\bigr],
\]

and fits a logistic regression:

\[
\Pr(y=1\mid \mathbf{s}) = \sigma(\mathbf{w}^\top \mathbf{s} + b).
\]

This is a discriminative baseline that can learn linear combinations of score features, unlike the hand-weighted `combined` score.

Notes on the `isolation_forest` option:

- The code supports it, but the current training call fits the forest on features without using labels (as is typical for one-class anomaly detection). This is mainly a convenience baseline and not the main path used in the notebook.

---

## 4) How the two-moons notebook uses the detector (end-to-end)

`notebooks/01_graph_manifold_two_moons.ipynb` follows this pipeline:

1. Train a classifier \(f_\theta\) on two moons.
2. Choose a reference space:
   - Input space (\(\phi(x)=x\)), or
   - Feature space (\(\phi(x)\) = penultimate activations). The notebook defaults to **feature space**.
3. Build the reference set \(Z_{\text{train}}\) and reference outputs \(f_{\text{train}}\).
4. For each evaluation point (clean/adv), compute scores via `compute_graph_scores`.
5. Concatenate clean and adversarial validation scores, label them, and fit:
   - `detector_type='score'`: estimate \(\tau\) as a clean percentile.
   - (Optionally) evaluate only **successful attacks** (those that actually flip the classifier’s prediction), which is important for small \(\epsilon\).
6. Evaluate on test by ranking using raw scores (AUC) and also by thresholding the detector’s probability proxy at 0.5.

Research note: AUC on raw scores is often the cleaner metric here, because it evaluates the ordering induced by the underlying statistic \(s(z)\) without conflating with a particular threshold or a non-calibrated sigmoid mapping.

---

## 5) Where “topology” fits (and what’s missing)

This repository is primarily **geometric/spectral** rather than explicitly topological:

- The graph Laplacian and diffusion operators are closely tied to the **intrinsic geometry** of \(M\) and can be viewed as capturing aspects of the manifold’s structure (including global connectivity) through spectral properties.
- However, the pipeline does **not** compute persistent homology, Reeb graphs, mapper complexes, or other explicit topological invariants.

If your research goal is topology-aware robustness, these scores are a strong baseline for **geometry-aware off-manifold detection**, and you can extend the framework by:

- computing persistent diagrams on neighborhood graphs (clean vs suspect),
- comparing local homology / Betti numbers across neighborhoods,
- using heat kernel signatures / spectral descriptors as topology-adjacent invariants,
- or testing topological stability under adversarial perturbations.

---

## 6) Practical notes + limitations (important for research claims)

- **Bandwidth / neighborhood dependence**: degree, Laplacian, diffusion, and tangent scores all depend on \(k\) and \(\sigma\) (explicitly or implicitly). The relevant asymptotic theorems require \(k\) / \(\sigma\) to scale with \(n\).
- **Feature space is a learned geometry**: when \(\phi\) is a neural representation, the “manifold” is model-dependent; adversarial perturbations can exploit representation folding.
- **Quantile calibration is distribution-specific**: FPR control holds only if clean deployment scores follow the same distribution as the clean validation set.
- **`predict_proba` is not calibrated**: the sigmoid around the threshold is a convenience mapping, not a probabilistic guarantee.

---

## 7) Pointers to the exact implementation

- Score computation: `src/graph_manifold.py`
  - `compute_degree_score`
  - `compute_laplacian_smoothness_score`
  - `compute_graph_scores` (adds tangent residuals and kNN radius; optional diffusion)
- Combined score: `src/compute_combined_score.py`
- Detector: `src/detectors.py`
  - `ScoreBasedDetector.fit`: sets threshold to a clean percentile
  - `ScoreBasedDetector.predict`: \(s(z)>\tau\)
  - `SupervisedGraphDetector`: logistic regression baseline
  - `train_graph_detector` / `predict_graph_detector`: orchestration and combined-score handling
- Notebook usage: `notebooks/01_graph_manifold_two_moons.ipynb`
  - sets `config.detector.score_type='tangent_residual_z'` and `detector_type='score'`

---

## 8) References (starting points)

These are canonical references underlying the theorems alluded to above:

- Belkin, M. & Niyogi, P. (2005/2007). Convergence of Laplacian eigenmaps / graph Laplacians to Laplace–Beltrami.
- Coifman, R. R. & Lafon, S. (2006). Diffusion maps.
- von Luxburg, U. (2007). A tutorial on spectral clustering (useful background on Laplacians and cuts).
- Singer, A. and related work on local PCA / tangent space estimation on manifolds (for formal convergence of tangent estimators).


