# Theory: Topological (Persistent Homology) Detector

This document focuses **only** on the repository’s *topological* detector: a scoring detector whose statistic is derived from **persistent homology (PH)** on local neighborhoods.

It is organized in two layers:
- **Part A (Conceptual)**: what the detector is doing and why it should work.
- **Part B (Mathematical)**: precise definitions of each object and the resulting score/decision rule.

---

## Part A — Conceptual explanation (what it does, at a high level)

### A1) What problem is the detector solving?

Given an input \(x\), we want a scalar **suspiciousness score** \(s(x)\) that is larger when \(x\) is likely adversarial/off-distribution relative to the clean training distribution.

In this repo, “off-manifold” is reinterpreted as **locally atypical structure** in a representation space: adversarial examples that flip the classifier often land in regions where the *local neighborhood structure* (connectivity/loops across scales) looks different than clean data.

### A2) What is the “topological signal”?

Instead of checking geometric distances directly (e.g., kNN radius), we compute **persistent homology** on a local neighborhood point cloud around the representation of \(x\). Persistent homology summarizes how:
- connected components merge (H0), and
- loop-like structures appear/disappear (H1),
as we thicken the neighborhood by a scale parameter \(\varepsilon\).

These summaries are “topological descriptors across scale.” In practice, they act as a signature of local structure that can shift under successful adversarial perturbations.

### A2.1 Is this method “data-agnostic”?

At the *data type* level (images vs tabular vs text embeddings), this method is largely **data-agnostic**: it requires only that each sample \(x\) can be mapped into a **metric space** where neighborhoods are meaningful.

More precisely, the detector assumes:
- there is a representation map \(\phi\) such that \(\phi(x)\) lies in a space where a distance \(d(\cdot,\cdot)\) captures local similarity,
- clean data are sufficiently sampled so local neighborhoods \(N_k(\phi(x))\) are stable,
- adversarial examples that *successfully* flip the classifier tend to land in regions where local structural/topological signatures differ.

So it is **not universally data-agnostic** in the sense of making no assumptions at all: it critically depends on the choice of \(\phi\), the metric \(d\), and the neighborhood scale \(k\).

### A3) What does the detector output?

For each \(x\), the pipeline is:

1. **Represent**: compute \(z=\phi(x)\) (either \(z=x\) or a neural embedding).
2. **Localize**: choose a local cloud \(P(z)\) from nearby training representations.
3. **Topologize**: compute PH diagrams \(D_0(z), D_1(z)\) for \(P(z)\).
4. **Vectorize**: map diagrams to a fixed-length feature vector \(v(x)\).
5. **Score**: measure how far \(v(x)\) is from the clean distribution (Mahalanobis distance).
6. **Decide**: flag if \(s(x)\) exceeds a percentile threshold set on clean validation data.

### A4) Why does local PCA before PH help (especially for high-D data)?

On datasets with many dimensions, the ambient dimension \(d\) can be large relative to the neighborhood size \(|P(z)|\). Two issues appear:
- distances concentrate (neighbors become nearly equidistant),
- VR PH becomes dominated by high-dimensional noise.

The repo therefore supports an **optional local PCA** step that projects each neighborhood cloud to a low dimension \(d'\) before PH. Conceptually, this attempts to recover the **local intrinsic structure** where topology is informative, while discarding nuisance directions.

---

## Part B — Mathematical explanation (definitions → score)

### B1) Representation space

We work with a representation map:

\[
z = \phi(x) \in \mathbb{R}^d.
\]

Let the clean reference set be

\[
Z_{\text{train}}=\{z_i\}_{i=1}^n = \{\phi(x_i)\}_{i=1}^n \subset \mathbb{R}^d.
\]

### B2) Local neighborhood point cloud

To avoid being tied to “point clouds” as a data modality, we formalize this step as operating on a **finite subset of a metric space**.

Let \((\mathcal{Z}, d)\) be a metric space and let \(\phi:\mathcal{X}\to\mathcal{Z}\) map inputs into \(\mathcal{Z}\). The clean reference set is the finite subset

\[
Z_{\text{train}} = \{\phi(x_i)\}_{i=1}^n \subset \mathcal{Z}.
\]

For a query point \(z=\phi(x)\), define its kNN neighborhood in \(Z_{\text{train}}\) (default: Euclidean metric when \(\mathcal{Z}=\mathbb{R}^d\)):

\[
N_k(z) = \{z_{(1)},\dots,z_{(k)}\}\subset Z_{\text{train}}.
\]

The local cloud used for PH is:

\[
P(z)=\{z\}\cup N_k(z).
\]

**Note:** distance is used only to define locality; the detector’s statistic is computed from PH-derived features.

#### Design choice motivation (B2)

- **Why kNN neighborhoods?** We need a *local* structure estimator that is simple and generic in any metric space. kNN adapts to non-uniform sampling density better than a fixed-radius ball (which can become too small in sparse regions and too large in dense regions).
- **Why include the query point \(z\) in \(P(z)\)?** It makes the computed topology explicitly about how \(z\) “attaches” to its local region across scales, rather than only describing the neighbors among themselves.
- **Why is this still a topology-based detector?** The neighborhood operator uses geometry only to define “local.” The detection statistic comes from PH features of the induced complex, not from a distance threshold on \(z\).

### B3) Optional preprocessing: local PCA projection (Option 1)

When enabled, we apply a local PCA map \(\Pi_z:\mathbb{R}^d\to\mathbb{R}^{d'}\) learned from the neighborhood cloud:

\[
\Pi_{z}(u)=W_z^\top u,\qquad W_z\in\mathbb{R}^{d\times d'}.
\]

In practice \(d'\) is bounded by the local rank of the cloud: \(d'\le \min(d,|P(z)|-1)\).

We then replace the cloud with its projection:

\[
\tilde{P}(z)=\Pi_z(P(z))\subset\mathbb{R}^{d'}.
\]

#### Design choice motivation (B3)

- **Why local PCA before PH?** VR PH is driven by pairwise distances. In high-dimensional regimes, distances can concentrate and become noisy, causing persistence summaries to collapse toward similar values for many points. Projecting each neighborhood to a lower-dimensional subspace often restores signal by emphasizing local intrinsic variation and suppressing nuisance directions.
- **Why *local* PCA (per neighborhood) instead of global PCA?** A single global projection can mix distinct regions/modes; local PCA adapts to local anisotropy/curvature and is closer to a local chart/tangent approximation.
- **Why cap \(d' \le |P(z)|-1\)?** The neighborhood cloud has rank at most \(|P(z)|-1\) after centering; projecting above that is ill-posed and adds no information.

### B4) Vietoris–Rips filtration

Given a finite metric space \((\tilde{P}(z), d)\), define the Vietoris–Rips complex at scale \(\varepsilon\ge 0\):

\[
\mathrm{VR}_\varepsilon(\tilde{P}(z))=\left\{\sigma\subseteq \tilde{P}(z): d(p,q)\le\varepsilon\ \forall p,q\in\sigma\right\}.
\]

As \(\varepsilon\) increases this forms a filtration:

\[
\mathrm{VR}_{\varepsilon_1}\subseteq \mathrm{VR}_{\varepsilon_2}\quad (\varepsilon_1\le\varepsilon_2).
\]

#### Design choice motivation (B4)

- **Why Vietoris–Rips (VR)?** VR is defined purely from distances, so it’s maximally compatible with the detector’s “metric-space” framing. It also has fast, widely used implementations (`ripser`) that make repeated local PH feasible.
- **Why not α/Čech/cubical by default?** α/Čech are more geometry-dependent (and most practical in low-dimensional Euclidean settings), and cubical complexes assume grid structure. VR is the simplest “works everywhere” choice for this repo’s experiments.

### B5) Persistent homology and diagrams

For each homology dimension \(k\in\{0,1\}\) (default), persistent homology produces a multiset of birth/death pairs:

\[
D_k(z)=\{(b_i,d_i)\}_{i=1}^{m_k},\quad d_i\ge b_i.
\]

Define lifetimes \(\ell_i=d_i-b_i\). Short lifetimes below a cutoff \(\ell_{\min}\) are treated as numerical noise.

#### Design choice motivation (B5)

- **Why compute only \(H_0\) and \(H_1\) (default)?** For small-to-moderate neighborhood sizes, \(H_0\) captures local connectivity/cluster merging and \(H_1\) captures loop-like structure, which are often the most interpretable and computationally tractable. Higher dimensions increase cost substantially and are frequently unstable in small local samples.
- **Why discard tiny lifetimes?** Very short-lived features are often numerical artifacts or sampling noise; filtering improves stability and reduces variance of downstream summaries.

### B6) Diagram \(\to\) feature vector

We construct a fixed-length vector \(v(x)\in\mathbb{R}^p\) from \(D_0(z)\) and \(D_1(z)\) using summary functionals. For each \(k\):

- Count: \(\text{count}_k = m\)
- Total persistence: \(\text{tp}_k = \sum_i \ell_i\)
- Max persistence: \(\text{max}_k = \max_i \ell_i\)
- L2 persistence: \(\text{l2}_k = \sqrt{\sum_i \ell_i^2}\)
- Persistence entropy:
\[
p_i=\frac{\ell_i}{\sum_j \ell_j},\qquad H_k=-\sum_i p_i\log p_i.
\]

The concatenation of these per-\(k\) summaries is the topology feature vector \(v(x)\).

#### Design choice motivation (B6)

- **Why summarize diagrams at all?** The detector needs a fixed-length vector \(v(x)\) for efficient scoring and calibration. Summary statistics are a lightweight baseline that still reflects multi-scale topology.
- **Why these particular summaries (count/total/max/L2/entropy)?** Together they capture “how many features exist,” “how strong/long-lived they are,” and “whether persistence is concentrated in a few dominant features vs spread across many.”
- **Why not persistence images/landscapes here?** Those can be stronger but introduce extra hyperparameters and compute; the repo starts with minimal, interpretable features and leaves richer embeddings as a natural extension.

### B7) Scoring via a (shrunk) Mahalanobis distance

Let \(\mathcal{C}\) be a set of clean calibration points. Estimate:

\[
\mu = \frac{1}{|\mathcal{C}|}\sum_{x\in\mathcal{C}} v(x),\qquad
\Sigma = \frac{1}{|\mathcal{C}|-1}\sum_{x\in\mathcal{C}} (v(x)-\mu)(v(x)-\mu)^\top.
\]

Apply diagonal shrinkage:

\[
\Sigma_\lambda=\Sigma+\lambda I,
\]

and define the score:

\[
s(x)=\sqrt{(v(x)-\mu)^\top \Sigma_\lambda^{\dagger}(v(x)-\mu)}.
\]

This is a natural anomaly score under a Gaussian model on \(v(x)\) and provides a single scalar suitable for thresholding and ROC evaluation.

#### B7.1 Is the score normalized?

In the current implementation, **the raw score \(s(x)\) is not normalized to a fixed range** like \([0,1]\).

- It is a distance in the topology-feature space, so its scale depends on the feature parameterization and covariance estimate.
- In practice, the score is made operational by comparing it to a threshold \(\tau\) calibrated on clean validation data.

However, there are two common “normalizations” you can use for interpretation:

1) **Threshold-normalized score**

\[
\tilde{s}(x) = \frac{s(x)}{\tau}.
\]

This makes \(\tilde{s}(x)>1\) equivalent to “flagged.”

2) **Chi-square / p-value style normalization (model-based)**

If \(v(x)\) were exactly Gaussian with covariance \(\Sigma_\lambda\) and dimension \(p\), then the squared distance

\[
s(x)^2 = (v(x)-\mu)^\top \Sigma_\lambda^{-1}(v(x)-\mu)
\]

would follow a \(\chi^2_p\) distribution under clean data (approximately). This motivates:
- using \(s(x)^2\) directly (monotone in \(s\)), or
- mapping to a p-value \(p(x)=1-F_{\chi^2_p}(s(x)^2)\) for interpretability.

We do **not** rely on this chi-square assumption for correctness (since the clean distribution in topology-feature space is typically not exactly Gaussian), but it can be useful for calibration/interpretation.

#### Design choice motivation (B7)

- **Why Mahalanobis distance?** It measures “atypicality” while accounting for feature scales and correlations, unlike raw Euclidean distance in feature space.
- **Why covariance shrinkage \(\Sigma_\lambda=\Sigma+\lambda I\)?** With finite clean calibration samples and correlated PH summaries, \(\Sigma\) can be ill-conditioned; shrinkage stabilizes inversion and makes scoring numerically robust.
- **Why a simple unsupervised score model?** It isolates the effect of topological features. Adding a learned detector head can increase power but also confounds whether the gain comes from topology or classifier capacity.

### B8) Thresholding by a clean quantile

Choose a target quantile \(q\in(0,1)\) (default \(q=0.95\)). Let \(\tau\) be the empirical \(q\)-quantile of \(s(x)\) on clean validation data. The decision rule is:

\[
\hat{y}(x)=\mathbf{1}\{s(x)>\tau\}.
\]

If the deployment clean distribution matches calibration, this targets an approximate false-positive rate of \(1-q\) (up to sampling error).

#### Design choice motivation (B8)

- **Why quantile thresholding?** It provides an interpretable false-positive control knob under the clean calibration distribution and avoids implying that \(s(x)\) is a calibrated probability.
- **Why not set \(\tau\) using adversarial samples?** Using adversarial data to choose \(\tau\) can leak attack specifics into calibration and can overfit to a particular threat model; clean-quantile thresholding stays threat-model-agnostic and supports ROC-style evaluation across attacks.

### B9) Stability perspective (why this can be robust)

PH satisfies stability guarantees: for many constructions,

\[
d_B(D_k(X),D_k(Y)) \lesssim d_H(X,Y),
\]

relating diagram distances to set perturbations (bottleneck vs Hausdorff). This motivates PH as a descriptor that should not be overly sensitive to small benign perturbations, while still changing when neighborhood structure shifts meaningfully.

We do **not** claim that PCA preprocessing preserves topology globally; it is a bias–variance tradeoff that can improve the signal-to-noise ratio of distance-based filtrations in high-dimensional regimes.

---

## Implementation mapping (topological detector only)

- **PH + summaries**: `src/topology_features.py`
  - `TopologyConfig(preprocess='none'|'pca', pca_dim=d')`
  - `local_persistence_features(...)` (does optional local PCA then calls PH backend)
  - `persistence_summary_features(...)` (count/total/max/L2/entropy)
- **Where features are computed for model inputs**: `src/graph_manifold.py`
  - `compute_graph_scores(..., use_topology=True)` emits `topo_h{0,1}_*` keys
  - controlled by `GraphConfig.topo_k`, `GraphConfig.topo_maxdim`, `GraphConfig.topo_preprocess`, `GraphConfig.topo_pca_dim`
- **Scoring detector**: `src/detectors.py`
  - `TopologyScoreDetector` (Mahalanobis score + percentile threshold)
  - enabled via `DetectorConfig.detector_type = 'topology_score'`

---

## Practical limitations (topological detector)

- **Compute**: PH per sample is expensive; neighborhood size \(k\) and feature dimension \(d'\) are the primary cost knobs.
- **High-dimensional regimes**: without local PCA, VR PH can be uninformative due to distance concentration; with PCA, \(d'\) must be tuned (too small collapses structure; too large reintroduces noise).
- **Adaptive attackers**: PH is typically non-differentiable; evaluation should include score-based black-box attacks to avoid false confidence from gradient obfuscation.
