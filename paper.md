---
title: 'ExpFamilyPCA.jl: A Julia Package for Exponential Family Principal Component Analysis'
tags:
  - Julia
  - compression
  - dimensionality reduction
  - PCA
  - exponential family
  - EPCA
  - open-source
  - POMDP
  - MDP
  - sequential decision making
  - RL
authors:
  - name: Logan Mondal Bhamidipaty
    orcid: 0009-0001-3978-9462
    affiliation: 1
  - name: Mykel J. Kochenderfer
    orcid: 0000-0002-7238-9663
    affiliation: 1
  - name: Trevor Hastie
    orcid: 0000-0002-0164-3142
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 9 September 2024
bibliography: paper.bib
---


# Summary

Principal component analysis (PCA) [@PCA] is an important tool for compression, interpretability, and denoising that works best when data is continuous, real, and normally distributed. Exponential family principal component analysis (EPCA) [@EPCA] generalizes PCA to accommodate observations from any exponential family, making it well-suited for binary, count, and discrete distribution data. `ExpFamilyPCA.jl` is the first Julia package [@Julia] to implement EPCA and the first package in any language to support EPCA with multiple distributions.

# Statement of Need

EPCA has not been widely adopted due to the lack of general-purpose implementations that support multiple distributions [@LitReview]; existing work often relies on custom optimization routines specific to a single distribution [@Roy; @MatlabEPCA]. This is unfortunate since Bregman-based methods have succeeded in applications like mass spectrometry [@spectrum], ultrasound denoising [@ultrasound], text analysis [@LitReview], and robust clustering [@clustering], suggesting that EPCA could be effectively applied in signal processing and machine learning.

Implementing a general-purpose EPCA algorithm is challenging in most languages because it requires symbolic differentiation and optimization, which often necessitates API-specific atoms that are not interoperable with optimization libraries in languages like Python or C. ExpFamilyPCA.jl overcomes these challenges by leveraging Julia's capabilities. Julia supports multiple dispatch, promoting high levels of code reuse [@dispatch], and allows symbolic manipulation [@symbolics], optimization [@optim], and numerically stable functions [@stable_exp] to work smoothly and efficiently together. This makes ExpFamilyPCA.jl a fast, numerically stable EPCA implementation that supports multiple distributions. It includes off-the-shelf models for most common distributions (§ [Supported Distributions](#supported-distributions)) and multiple constructors for custom distributions (§ [Custom Distributions](#supported-distributions)).

# Problem Formulation

## Principal Component Analysis

### Geometric Interpretation

Given a data matrix $X \in \mathbb{R}^{n \times d}$, the geometric goal of PCA is to find the closest low-rank approximation $\Theta \in \mathbb{R}^{n \times d}$. Formally,

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \frac{1}{2}\|X - \Theta\|_F^2 \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}$$

where $\| \cdot \|_F$ denotes the Frobenius norm.

### Probabilistic Interpretation

Since this objective is equivalent to maximizing the Gaussian log-likelihood, PCA can be viewed as a denoising procedure to recover a low-dimensional latent structure from high-dimensional noisy observations. Formally,

$$
x_i \sim \mathcal{N}(\theta_i, I)
$$

where $x_i$ is a row of $X$, $\theta_i$ is a row of $\Theta$, and $i \in \{1, \dots, n\}$.


## Bregman Divergences

Bregman divergences [@Bregman] provide a flexible measure of statistical difference. For a strictly convex and continuously differentiable function $F$, the Bregman divergence between $p, q \in \mathrm{dom}(F)$ is

$$
B_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle.
$$

When $F$ is chosen to be the convex conjugate of the log partition of an exponential family distribution, minimizing the induced Bregman divergence is the same as maximizing the correspodning log-likelihood [@azoury; @forster] (see [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/bregman/)).

## Exponential Family Principal Component Analysis

EPCA extends PCA by replacing its geometric objective with a probabilistic one that minimizes a generalized Bregman divergence:

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F(X \| g(\Theta)) + \epsilon B_F(\mu_0 \| g(\Theta)) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k.
\end{aligned}$$

In this formulation,

* $g(\theta)$ is the **link function** and the derivative of $G$,
* $F(\mu)$ is the **convex conjugate** or dual of $G$,
* $B_F(p \| q)$ is the **Bregman divergence** induced from $F$,
* and both $\mu_0 \in \mathrm{range}(g)$ and $\epsilon > 0$ are regularization hyperparameters.

This framework allows EPCA to perform dimensionality reduction on data from any exponential family distribution, broadening PCA's applicability beyond Gaussian data. Notably, PCA is a special case of EPCA when data follows a Gaussian distribution (see [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/gaussian/)). EPCA is analogous to how generalized linear models (GLMs) [@GLM] extend linear regression to accommodate various response distributions.

### Example: Poisson EPCA

For the Poisson distribution, the EPCA objective becomes the generalized Kullback-Leibler (KL) divergence (see [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/poisson/)), making Poisson EPCA ideal for compressing discrete distribution data. This is crucial in applications like belief compression in reinforcement learning [@Roy], where high-dimensional belief states can be effectively reduced with minimal information loss. Below we recreate a figure from @shortRoy and observe that Poisson EPCA achieved a nearly perfect reconstruction of a $41$-dimensional belief profile using just $5% basis components.

![](./scripts/kl_divergence_plot.png)

# API 

## Usage

Each `EPCA` object supports a three-method interface: `fit!`, `compress`, and `decompress`. `fit!` trains the model and returns the compressed training data; `compress` returns compressed input; and `decompress` reconstructs the original data from the compressed representation.

```julia
X = rand(n1, indim) * 100
Y = rand(n2, indim) * 100

X_compressed = fit!(gamma_epca, X)
Y_compressed = compress(gamma_epca, Y)
Y_reconstructed = decompress(gamma_epca, Y_compressed)
```

## Supported Distributions

`ExpFamilyPCA.jl` includes efficient EPCA implementations for several exponential family distributions.

| Julia                     | Description                                            |
|---------------------------|--------------------------------------------------------|
| `BernoulliEPCA` | For binary data                                        |
| `BinomialEPCA` | For count data with a fixed number of trials           |
| `ContinuousBernoulliEPCA` | For modeling probabilities between $0$ and $1$         |
| `GammaEPCA` | For positive continuous data                           |
| `GaussianEPCA` | Standard PCA for real-valued data                      |
| `NegativeBinomialEPCA` | For over-dispersed count data                          |
| `ParetoEPCA` | For modeling heavy-tailed distributions                |
| `PoissonEPCA` | For count and discrete distribution data               |
| `WeibullEPCA` | For modeling life data and survival analysis           |

## Custom Distributions

When working with custom distributions, certain specifications are often more convenient and computationally efficient than others. For example, inducing the gamma EPCA objective from the log-partition $G(\theta) = -\log(-\theta)$ and its derivative $g(\theta) = -1/\theta$ is much simpler than implementing the full the Itakura-Saito distance [@ItakuraSaito] (see [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/gamma/)):

$$
D(P(\omega), \hat{P}(\omega)) =\frac{1}{2\pi} \int_{-\pi}^{\pi} \Bigg[ \frac{P(\omega)}{\hat{P}(\omega)} - \log \frac{P(\omega)}{\hat{P}{\omega}} - 1\Bigg] \, d\omega.
$$

In `ExpFamilyPCA.jl`, we would write:

```julia
G(θ) = -log(-θ)
g(θ) = -1 / θ
gamma_epca = EPCA(indim, outdim, G, g, Val((:G, :g)); options = NegativeDomain())
```

A lengthier discussion of the `EPCA` constructors and math is provided in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/objectives/).

# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References