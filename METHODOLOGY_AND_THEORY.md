# Methodology and Theory: Graph/Laplacian Manifold Methods for Adversarial Detection

## Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
2. [Why This Method Should Work](#why-this-method-should-work)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Implementation Methodology](#implementation-methodology)
5. [Connection to Adversarial Detection](#connection-to-adversarial-detection)

---

## Theoretical Foundations

### 1. The Manifold Hypothesis

The **manifold hypothesis** states that high-dimensional data typically lies on or near a low-dimensional manifold embedded in the high-dimensional space. For natural data distributions, this manifold represents the "true" data structure.

**Formal statement**: Given a dataset $\mathcal{D} = \{x_i\}_{i=1}^n \subset \mathbb{R}^d$, there exists a low-dimensional manifold $\mathcal{M} \subset \mathbb{R}^d$ with intrinsic dimension $m \ll d$ such that:
$$
P(x \in \mathcal{M}) \approx 1 \quad \text{for } x \sim p_{\text{data}}
$$

**Implications for adversarial examples**:
- Clean examples: $x_{\text{clean}} \in \mathcal{M}$ (or very close)
- Adversarial examples: $x_{\text{adv}} = x_{\text{clean}} + \delta$, where $\delta$ is a small perturbation that pushes $x_{\text{adv}}$ off the manifold: $x_{\text{adv}} \notin \mathcal{M}$

### 2. Graph Laplacian Theory

#### 2.1 Graph Construction

Given training data $\mathcal{D}_{\text{train}} = \{z_i\}_{i=1}^n \subset \mathbb{R}^d$, we construct a **k-nearest neighbor (k-NN) graph** $G = (V, E)$ where:
- Vertices: $V = \{1, 2, \ldots, n\}$ (one per training point)
- Edges: $(i,j) \in E$ if $z_j$ is among the $k$ nearest neighbors of $z_i$ (or vice versa)

**Weight matrix** $W \in \mathbb{R}^{n \times n}$:
$$
W_{ij} = \begin{cases}
\exp\left(-\frac{\|z_i - z_j\|^2}{2\sigma^2}\right) & \text{if } (i,j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

where $\sigma$ is a scale parameter (typically set to the median pairwise distance).

**Degree matrix** $D$:
$$
D_{ii} = \sum_{j=1}^n W_{ij}, \quad D_{ij} = 0 \text{ for } i \neq j
$$

#### 2.2 Graph Laplacian

The **unnormalized graph Laplacian** is:
$$
L = D - W
$$

The **symmetric normalized Laplacian** is:
$$
L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}
$$

**Key properties**:
- $L$ is positive semi-definite
- The smallest eigenvalue is 0 with eigenvector $\mathbf{1}$ (constant function)
- Eigenvalues $\lambda_1 = 0 \leq \lambda_2 \leq \cdots \leq \lambda_n$ encode graph structure
- Small eigenvalues correspond to smooth functions on the graph

#### 2.3 Dirichlet Energy

For a function $f: V \to \mathbb{R}$ defined on the graph vertices, the **Dirichlet energy** is:
$$
E(f) = \frac{1}{2} \sum_{i,j} W_{ij} (f_i - f_j)^2 = f^T L f
$$

**Interpretation**:
- Measures how much $f$ varies across the graph
- Low energy: $f$ is smooth (similar values on connected vertices)
- High energy: $f$ varies rapidly (large differences between neighbors)

**Connection to smoothness**: A function with low Dirichlet energy is "smooth" with respect to the graph structure, meaning it respects the manifold geometry.

---

## Why This Method Should Work

### 1. Manifold Conformity Principle

**Core hypothesis**: Points that lie on or near the data manifold should:
1. Be well-connected to training points (high degree in k-NN graph)
2. Have model predictions that are smooth with respect to the graph structure (low Dirichlet energy)

**Adversarial examples violate this**:
- They are pushed off the manifold by small perturbations
- They have lower connectivity to training points
- Their predictions disrupt the smoothness of the model's output function

### 2. Mathematical Justification

#### 2.1 Connectivity Argument

For a point $z$ on the manifold $\mathcal{M}$:
- It lies in a dense region of training data
- Many training points are nearby
- High degree: $\deg(z) = \sum_{j \in \mathcal{N}_k(z)} W(z, z_j)$ is large

For an adversarial point $z_{\text{adv}} = z + \delta$ off the manifold:
- Fewer training points nearby
- Lower degree: $\deg(z_{\text{adv}}) < \deg(z)$
- **Degree score**: $s_{\text{deg}}(z) = -\deg(z)$ (higher = more suspicious)

#### 2.2 Smoothness Argument

Let $f: \mathbb{R}^d \to [0,1]$ be the model's output function (e.g., probability of class 1).

**On-manifold smoothness**: If $f$ is smooth on $\mathcal{M}$, then for nearby points $z_i, z_j$:
$$
|f(z_i) - f(z_j)| \approx \|\nabla_{\mathcal{M}} f\| \cdot d_{\mathcal{M}}(z_i, z_j)
$$
where $d_{\mathcal{M}}$ is the geodesic distance on the manifold.

**Dirichlet energy increment**: For a new point $z$ with neighbors $\mathcal{N}_k(z)$:
$$
\Delta E(z) = \sum_{j \in \mathcal{N}_k(z)} W(z, z_j) \cdot (f(z) - f(z_j))^2
$$

**For on-manifold points**:
- $f(z) \approx f(z_j)$ for nearby neighbors
- $\Delta E(z)$ is small

**For off-manifold (adversarial) points**:
- $f(z_{\text{adv}})$ may differ significantly from $f(z_j)$ even for nearby neighbors
- $\Delta E(z_{\text{adv}})$ is large
- **Laplacian score**: $s_{\text{lap}}(z) = \Delta E(z)$ (higher = more suspicious)

### 3. Theoretical Guarantees (Informal)

Under reasonable assumptions:
1. **Manifold regularity**: The data manifold is smooth and well-sampled
2. **Model smoothness**: The trained model $f$ is approximately smooth on the manifold
3. **Adversarial perturbation**: Adversarial examples are pushed off-manifold

Then:
- On-manifold points have **high degree** and **low Dirichlet energy increment**
- Off-manifold points have **low degree** and **high Dirichlet energy increment**
- The scores provide a **separable signal** for detection

---

## Mathematical Formulation

### 1. Graph Construction

**Input**: Training data $\{z_i\}_{i=1}^n \subset \mathbb{R}^d$ (in input or feature space)

**Step 1**: Build k-NN graph
$$
\mathcal{N}_k(z_i) = \{j : z_j \text{ is among } k \text{ nearest neighbors of } z_i\}
$$

**Step 2**: Compute edge weights
$$
W_{ij} = \begin{cases}
\exp\left(-\frac{\|z_i - z_j\|^2}{2\sigma^2}\right) & \text{if } j \in \mathcal{N}_k(z_i) \text{ or } i \in \mathcal{N}_k(z_j) \\
0 & \text{otherwise}
\end{cases}
$$

**Step 3**: Make symmetric (undirected graph)
$$
W \leftarrow \frac{W + W^T}{2}
$$

**Step 4**: Compute degree matrix
$$
D_{ii} = \sum_{j=1}^n W_{ij}
$$

### 2. Degree Score

For a new point $z \in \mathbb{R}^d$:

**Step 1**: Find k nearest neighbors in training data
$$
\mathcal{N}_k(z) = \{j : z_j \text{ is among } k \text{ nearest neighbors of } z\}
$$

**Step 2**: Compute edge weights to neighbors
$$
w_j = \exp\left(-\frac{\|z - z_j\|^2}{2\sigma^2}\right), \quad j \in \mathcal{N}_k(z)
$$

**Step 3**: Compute degree (total connectivity)
$$
\deg(z) = \sum_{j \in \mathcal{N}_k(z)} w_j
$$

**Step 4**: Return negative degree as score
$$
s_{\text{deg}}(z) = -\deg(z)
$$

**Interpretation**: Higher score = lower connectivity = more suspicious (off-manifold)

### 3. Laplacian Smoothness Score

For a new point $z$ with model output $f(z)$:

**Step 1**: Find k nearest neighbors $\mathcal{N}_k(z)$ and their model outputs $\{f(z_j)\}_{j \in \mathcal{N}_k(z)}$

**Step 2**: Compute edge weights
$$
w_j = \exp\left(-\frac{\|z - z_j\|^2}{2\sigma^2}\right), \quad j \in \mathcal{N}_k(z)
$$

**Step 3**: Compute Dirichlet energy increment
$$
\Delta E(z) = \sum_{j \in \mathcal{N}_k(z)} w_j \cdot (f(z) - f(z_j))^2
$$

**Step 4**: Return as score
$$
s_{\text{lap}}(z) = \Delta E(z)
$$

**Interpretation**: Higher score = more disruption to smoothness = more suspicious (off-manifold)

### 4. Combined Score (Optional)

For multiple scores, we can combine them:
$$
s_{\text{combined}}(z) = \alpha \cdot \bar{s}_{\text{deg}}(z) + \beta \cdot \bar{s}_{\text{lap}}(z)
$$

where $\bar{s}$ denotes normalized scores (e.g., z-score normalization) and $\alpha + \beta = 1$.

---

## Implementation Methodology

### 1. Data Preparation

**Input**: Training data $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^n$

**Representation selection**:
- **Input space**: $z_i = x_i$ (raw features)
- **Feature space**: $z_i = \phi(x_i)$ where $\phi$ extracts hidden layer representations

**Model outputs**: Compute $f_i = f(x_i)$ for all training points (e.g., probability of class 1)

### 2. Graph Construction Algorithm

```python
def build_knn_graph(Z_train, k, sigma=None):
    """
    Build k-NN graph with Gaussian edge weights.
    
    Steps:
    1. Find k nearest neighbors for each point using Euclidean distance
    2. If sigma not provided, set sigma = median(pairwise_distances)
    3. Compute Gaussian weights: W_ij = exp(-||z_i - z_j||² / (2σ²))
    4. Make symmetric: W = (W + W^T) / 2
    5. Compute degree matrix: D_ii = sum_j W_ij
    """
```

**Complexity**: $O(n^2 \log k)$ for k-NN search, $O(n^2)$ for weight computation

### 3. Score Computation Algorithm

#### 3.1 Degree Score

```python
def compute_degree_score(z, Z_train, k, sigma):
    """
    For a new point z:
    1. Find k nearest neighbors in Z_train
    2. Compute distances to neighbors
    3. Compute Gaussian weights: w_j = exp(-dist_j² / (2σ²))
    4. Return: -sum(w_j)  (negative for higher = more suspicious)
    """
```

#### 3.2 Laplacian Smoothness Score

```python
def compute_laplacian_smoothness_score(z, f_z, Z_train, f_train, k, sigma):
    """
    For a new point z with model output f_z:
    1. Find k nearest neighbors in Z_train
    2. Get their model outputs f_train[neighbors]
    3. Compute weights: w_j = exp(-dist_j² / (2σ²))
    4. Compute Dirichlet energy increment:
       ΔE = sum_j w_j * (f_z - f_train[j])²
    5. Return ΔE
    """
```

### 4. Batch Processing

For efficiency, we process multiple points:

```python
def compute_graph_scores(X_points, model, Z_train, f_train, graph_params):
    """
    Compute scores for a batch of points.
    
    Steps:
    1. Extract representations Z_points (input or feature space)
    2. Get model outputs f_points for all points
    3. For each point:
       - Compute degree score
       - Compute Laplacian smoothness score
    4. Return dictionary of score arrays
    """
```

**Optimization**: Can be parallelized across points since score computation is independent.

### 5. Detector Training

**Score-based detector**:
1. Compute scores for validation set (clean + adversarial)
2. Set threshold at percentile of clean scores: $\tau = \text{percentile}(s_{\text{clean}}, p)$
3. Predict: $\hat{y} = \mathbb{1}[s(z) > \tau]$

**Supervised detector**:
1. Use scores as features: $\mathbf{x} = [s_{\text{deg}}(z), s_{\text{lap}}(z)]$
2. Train classifier (e.g., logistic regression) on labeled data
3. Predict: $\hat{y} = \text{classifier}(\mathbf{x})$

### 6. Error Probability Calibration

**Goal**: Map scores to error probabilities $P(\text{error} | s)$

**Methods**:
- **Isotonic regression**: Non-parametric, monotonic mapping
- **Logistic regression**: Parametric, learns $P(\text{error}) = \sigma(\alpha s + \beta)$

**Calibration metrics**:
- **ECE (Expected Calibration Error)**: Average difference between predicted and actual error rates
- **MCE (Max Calibration Error)**: Maximum difference in any bin

---

## Connection to Adversarial Detection

### 1. Theoretical Link

**Manifold hypothesis** → Adversarial examples are off-manifold → Detectable via manifold conformity

**Graph Laplacian** → Captures local manifold structure → Provides smoothness measure

**Dirichlet energy** → Measures prediction smoothness → High energy indicates off-manifold points

### 2. Empirical Validation

From experimental results:
- **Score separation**: Clean examples have lower scores than adversarial examples
- **ROC AUC = 0.7834**: Good separability between clean and adversarial
- **Mean scores**: 
  - Clean: $\mu_{\text{clean}} = 0.0176$, $\sigma_{\text{clean}} = 0.1978$
  - Adversarial: $\mu_{\text{adv}} = 0.1708$, $\sigma_{\text{adv}} = 0.6313$

### 3. Advantages

1. **Unsupervised component**: Graph construction doesn't require adversarial examples
2. **Interpretable**: Scores have clear geometric meaning
3. **Flexible**: Works in input or feature space
4. **Efficient**: k-NN graph is sparse, computation is local

### 4. Limitations and Assumptions

1. **Manifold assumption**: Requires data to lie on a low-dimensional manifold
2. **Smoothness assumption**: Model predictions should be smooth on the manifold
3. **Sampling assumption**: Training data should be dense enough to capture manifold structure
4. **Hyperparameter sensitivity**: Choice of $k$ and $\sigma$ affects performance

### 5. Extensions

- **Diffusion maps**: Spectral embedding for better manifold representation
- **Multi-scale**: Use multiple values of $k$ or $\sigma$
- **Feature space**: May capture learned representations better than input space
- **Combined scores**: Leverage multiple signals for better detection

---

## Summary

This method leverages the **manifold hypothesis** and **graph Laplacian theory** to detect adversarial examples by measuring their **conformity to the data manifold**. The approach is theoretically grounded, computationally efficient, and empirically validated. The key insight is that adversarial examples, being off-manifold, exhibit:
1. **Low connectivity** (degree score)
2. **High prediction disruption** (Laplacian smoothness score)

These signals provide a principled way to distinguish clean from adversarial examples without requiring adversarial training data.

