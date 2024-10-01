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

Principal component analysis (PCA) [@PCA] is a widely used tool for compressing, interpreting, and denoising data that works best with Gaussian data. Exponential family principal component analysis (EPCA) [@EPCA] generalizes PCA to handle data from any exponential family, making it more appropriate for binary, count, and probability data common in science and machine learning. `ExpFamilyPCA.jl` is the first Julia package [@Julia] to implement EPCA and the first in any language to support multiple distributions for EPCA.

# Statement of Need

The limited adoption of EPCA likely stems from a lack of available tools, with the only existing package supporting just a single distribution [@epca-MATLAB]. This is surprising given that other Bregman-based optimization techniques have been successful in fields such as mass spectrometry [@spectrum], ultrasound denoising [@ultrasound], text analysis [@LitReview], and robust clustering [@clustering]. These successes suggest that EPCA has untapped potential in signal processing and machine learning.

The primary reason no general EPCA library exists may be the difficulty of implementation in most programming languages. In Python and C, for example, symbolic differentiation and optimization libraries are not typically interoperable. Julia, by contrast, uses multiple dispatch which facilitates high levels of generic code reuse [@dispatch]. Multiple dispatch allows `ExpFamilyPCA.jl` to integrate fast symbolic differentiation [@symbolics], optimization [@optim], and numerically stable computation [@stable_exp] without requiring costly API conversions. As a result, `ExpFamilyPCA.jl` delivers speed, stability, and flexibility, with built-in support for most common distributions (§ [Supported Distributions](#supported-distributions)) and flexible constructors for custom distributions (§ [Custom Distributions](#supported-distributions)).

# Problem Formulation

## Principal Component Analysis

### Geometric Interpretation

Given a data matrix $X \in \mathbb{R}^{n \times d}$, the goal of PCA is to find the best low-rank approximation $\Theta \in \mathbb{R}^{n \times d}$. Formally, this can be expressed as:

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \frac{1}{2}\|X - \Theta\|_F^2 \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}$$

where $\| \cdot \|_F$ denotes the Frobenius norm.

### Probabilistic Interpretation

This objective also maximizes the Gaussian log-likelihood, meaning PCA can also be viewed as a method to recover a low-dimensional structure from high-dimensional, noisy data. Specifically, for each data point $x_i$ (a row of $X$):

$$
x_i \sim \mathcal{N}(\theta_i, I)
$$

where $\theta_i$ (a row of $\Theta$) is the mean. The rank constraint $k$ in the geometric interpretation now corresponds to the parameter space

$$\begin{aligned}
& \underset{\Theta}{\text{maximize}}
& & \sum_{i=1}^{n}\log \mathcal{L}(x_i; \theta_i) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}$$

where $\mathcal{L}$ is the likelihood function.

## Bregman Divergences

Bregman divergences [@Bregman] provide a way to measure statistical differences. For a strictly convex and continuously differentiable function $F$, the Bregman divergence between between $p, q \in \mathrm{dom}(F)$ is

$$
B_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle.
$$

When $F$ is chosen to be the convex conjugate of the log-partition of an exponential family distribution, minimizing the induced Bregman divergence is the same as maximizing the corresponding log-likelihood [@azoury; @forster] (see [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/bregman/)). Since the Gaussian distribution is in the exponential family, this means that Bregman divergences generalize the PCA objective (see [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/gaussian/)).

## Exponential Family Principal Component Analysis

EPCA extends PCA by replacing its geometric objective with a probabilistic one that minimizes a generalized Bregman divergence:

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F(X \| g(\Theta)) + \epsilon B_F(\mu_0 \| g(\Theta)) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k.
\end{aligned}$$

Here

* $g(\theta)$ is the **link function** and the derivative of $G$,
* $G(\theta)$ is an arbitrary convex, differentiable function (usually the **log-parition** of an exponential family distribution),
* $F(\mu)$ is the **convex conjugate** or dual of $G$,
* $B_F(p \| q)$ is the **Bregman divergence** induced from $F$,
* and both $\mu_0 \in \mathrm{range}(g)$ and $\epsilon > 0$ are regularization terms.

EPCA can handle data from any exponential family distribution, making it more versatile than PCA. PCA is a special case of EPCA when applied to Gaussian data, similar to how generalized linear models [@GLM] extend linear regression to different types of data distributions.

### Example: Poisson EPCA

For the Poisson distribution, the EPCA objective becomes the generalized Kullback-Leibler (KL) divergence (see [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/poisson/)), making Poisson EPCA ideal for compressing discrete distribution data. This is useful in applications like belief compression in reinforcement learning [@Roy], where high-dimensional belief states can be effectively reduced with minimal information loss. Below we recreate a figure from @shortRoy and observe that Poisson EPCA achieved a nearly perfect reconstruction of a $41$-dimensional belief profile using just $5$ basis components.

![](./scripts/kl_divergence_plot.png)

# API 

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

## Usage

Each `EPCA` object supports a three-method interface: `fit!`, `compress`, and `decompress`. `fit!` trains the model and returns the compressed training data; `compress` returns compressed input; and `decompress` reconstructs the original data from the compressed representation.

```julia
X = sample_from_gamma(n1, indim)
Y = sample_from_gamma(n2, indim)

X_compressed = fit!(gamma_epca, X)
Y_compressed = compress(gamma_epca, Y)
Y_reconstructed = decompress(gamma_epca, Y_compressed)
```

# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References