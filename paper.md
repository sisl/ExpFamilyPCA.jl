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

Principal component analysis (PCA) [@PCA] is a fundamental tool in data science and machine learning for dimensionality reduction and denoising. While PCA is effective for continuous, real-valued data, it may not perform well for binary, count, or discrete distribution data. Exponential family PCA (EPCA) [@EPCA] generalizes PCA to accommodate these data types, making it more suitable for tasks such as belief compression in reinforcement learning [@Roy]. `ExpFamilyPCA.jl` is the first Julia [@Julia] package for EPCA, offering fast implementations for common distributions and a flexible interface for custom distributions.

# Statement of Need

<!-- REDO -->

To our knowledge, there are no open-source implementations of EPCA, and the sole proprietary package [@epca-MATLAB] is limited to a single distribution. Modern applications of EPCA in reinforcement learning [@Roy] and mass spectrometry [@spectrum] require multiple distributions, numerical stability, and the ability to handle large datasets. `ExpFamilyPCA.jl` addresses this gap by providing fast implementations for several exponential family distributions and multiple constructors for custom distributions. More implementation and mathematical details are in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/).

# Problem Formulation

- PCA has a specific geometric objective in terms of projections
- This can also be interpreted as a denoising process using Gaussian MLE
- EPCA generalizes geometric objective using Bregman divergences which are related to exponential families

TODO: read the original GLM paper

PCA has many interpretations (e.g., a variance-maximizing compression, a distance-minimizing projection). The interpretation that is most useful for understanding EPCA is the denoising interpretration. Suppose we have $n$ noisy observations $x_1, \dots, x_n \in \mathbb{R}^{n \times d$

## Principal Component Analysis





Traditional PCA is a low-rank matrix approximation problem. For a data matrix $X \in \mathbb{R}^{n \times d}$ with $n$ observations, we want to find the low-rank matrix approximation $\Theta \in \mathbb{R}^{n \times d}$ such that $\mathrm{rank}(\Theta) = k \leq d$. Formally,

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \|X - \Theta\|_F^2 \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}$$

where $\| \cdot \|_F$ denotes the Frobenius norm[^1] and $\Theta = AV$ where $A = X_k \in \mathbb{R}^{n \times k}$ and $V = X_k \in \mathbb{R}^{k \times d}$.

[^1]: The Frobenius norm is a generalization of the Euclidean distance and thus a special case of the Bregman divergence (induced from the log-partition of the normal distribution).

## Exponential Family PCA

EPCA is a generalization of PCA that replaces PCA's geometric objective with a more general probabilistic objective that minimizes the generalized Bregman divergence—a measure closely related to the exponential family (see [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/bregman/))—rather than the squared Frobenius norm. The Bregman divergence $B_F$ associated with $F$ is defined [@Bregman]:

$$
B_F(p, q) = F(p) - F(q) - \nabla F(q) \cdot (p - q).
$$

The Bregman-based objective makes EPCA particularly versatile for dimensionality reduction when working with non-Gaussian data distributions:

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F(X \| g(\Theta)) + \epsilon B_F(\mu_0 \| g(\Theta)) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k.
\end{aligned}$$

In this formulation,

*  $g(\theta)$ is the **link function** and the derivative of $G$,
*  $F(\mu)$ is the **convex conjugate** or dual of $G$,
*  $B_F(p \| q)$ is the **Bregman divergence** induced from $F$,
*  and both $\mu_0 \in \mathrm{range}(g)$ and $\epsilon > 0$ are regularization hyperparameters.

EPCA is similar to generalized linear models (GLMs) [@GLM]. Just as GLMs extend linear regression to handle a variety of response distributions, EPCA generalizes PCA to accommodate data with noise drawn from any exponential family distribution, rather than just Gaussian noise. This allows EPCA to address a broader range of real-world data scenarios where the Gaussian assumption may not hold (e.g., binary, count, discrete distribution data).

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
| `BernoulliEPCA`           | For binary data                                        |
| `BinomialEPCA`            | For count data with a fixed number of trials           |
| `ContinuousBernoulliEPCA` | For modeling probabilities between $0$ and $1$         |
| `GammaEPCA`               | For positive continuous data                           |
| `GaussianEPCA`            | Standard PCA for real-valued data                      |
| `NegativeBinomialEPCA`    | For over-dispersed count data                          |
| `ParetoEPCA`              | For modeling heavy-tailed distributions                |
| `PoissonEPCA`             | For count and discrete distribution data               |
| `WeibullEPCA`             | For modeling life data and survival analysis           |

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

## Applications

The practical applications of `ExpFamilyPCA.jl` span several domains that deal with non-Gaussian data. One notable example is in reinforcement learning, specifically in belief state compression for partially observable Markov decision processes (POMDPs). Using Poisson EPCA, the package effectively reduces high-dimensional belief spaces with minimal information loss, as demonstrated by recreating results from @shortRoy. In this case, Poisson EPCA achieved nearly perfect reconstruction of a $41$-dimensional belief profile using just five basis components [CITE `CompressedBeleifMDPS.jl`, PAPER IN PRE-REVIEW]. Poisson EPCA compression also produces a more interpretable compression than traditional PCA because it minimizes the generalized KL divergence rather than the Frobenius norm.

![](./scripts/kl_divergence_plot.png)

`ExpFamilyPCA.jl` can also be used in fields like mass spectrometry and survival analysis, where specific data distributions like the gamma or Weibull may be more appropriate. By minimizing divergences suited to the distribution, `ExpFamilyPCA.jl` provides more accurate and interpretable dimensionality reduction than standard PCA.

# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References