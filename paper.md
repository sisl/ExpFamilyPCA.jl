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

Principal component analysis (PCA) [@PCA] is an important tool for compression, interpretability, and denoising that works best when data is continuous, real, and normally distributed. Exponential family principal component analysis (EPCA) [@EPCA] generalizes PCA to accommodate observations from any exponential family, making it well-suited for working with binary, count, and discrete distribution data. `ExpFamilyPCA.jl` is the first Julia package [@Julia] to implement EPCA and the first package in any language to support EPCA with multiple distributions.

# Statement of Need

`ExpFamilyPCA.jl` is a fast, numerically stable package for compressing, interpreting, and denoising high-dimensional data from exponential family distributions. It includes off-the-shelf models for most common distributions (§ [Supported Distributions](#supported-distributions)) and multiple constructors for custom distributions (§ [Custom Distributions](#supported-distributions)).

EPCA has not been widely adopted due to the lack of general-purpose implementations for multiple distributions, and the limited existing work relies on custom optimization routines that are specific to a single distribution [@Roy, MatlabEPCA]. This is unfortunate since Bregman-based methods have succeeded in mass spectrometry [@spectrum], ultrasound denoising [@ultrasound], text analysis [@LitReview], and robust clustering [@clustering], indicating EPCA’s broader potential in signal processing and machine learning.

Developing a general EPCA package is challenging, because defining the EPCA objective requires per-distribution implementations, symbolic differentiation, and optimization.In languages like Python and R, these functionalities are provided by specific libraries that use unique atoms, which may not integrate well with other packages. In contrast, Julia's support for multiple dispatch promotes high code reuse and package interoperability [@dispatch]. ExpFamilyPCA.jl leverages these strengths of Julia, along with its innate support for high-performance scientific computing, to enable fast symbolic differentiation [@symbolics], optimization [@optim], and numerically stable exponential operations [@stable_exp]. This makes it a fast, numerically stable EPCA implementation that supports multiple distributions.

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

Observe that the PCA objective is equivalent to maximizing the Gaussian log-likelihood, so PCA can be viewed as a denoising procedure that recovers a latent, low-dimensional  structure from high-dimensional observations corrupted with Gaussian noise. Formally,

$$
x_i \sim \mathcal{N}(\theta_i, I)
$$

where $x_i$ is a row of $X$, $\theta_i$ is a row of $\Theta$, and $i \in \{1, \dots, n\}$.


## Bregman Divergences

Bregman divergences [@Bregman] provide a flexible measure of statistical difference. For a strictly convex and continuously differentiable function $F$, the Bregman divergence between $p, q \in \mathrm{dom}(F)$ is

$$
B_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle.
$$

When $F$ is chosen to be the convex conjugate of the log partition of an exponential family distribution, minimizing the Bregman divergence aligns with maximizing the log-likelihood [@azoury, @forster]. This property makes Bregman divergences particularly suitable for extending PCA to the exponential family. A comprehensive discussion can be found in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/bregman/).

## Exponential Family Principal Component Analysis

EPCA extends the classical PCA by replacing its geometric objective with a probabilistic one that minimizes a generalized Bregman divergence instead of the squared Frobenius norm. Formally, EPCA seeks to solve:

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


This objective allows EPCA to effectively perform dimensionality reduction on data drawn from any exponential family distribution, thereby broadening the applicability of PCA beyond Gaussian data. Notably, PCA itself is a special case of EPCA when the data follows a Gaussian distribution, as detailed in the [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/gaussian/).

EPCA shares similarities with generalized linear models (GLMs) [@GLM]. Just as GLMs extend linear regression to accommodate various response distributions, EPCA generalizes PCA to handle data with noise from any exponential family distribution, not limited to Gaussian noise. This generalization allows EPCA to better model real-world data scenarios where the Gaussian assumption may not hold, such as binary, count, or other discrete distributions.

### Example: Poisson EPCA

For the Poisson distribution, the associated EPCA objective is the generalized Kullback-Leibler (KL) divergence (see [appendix](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/appendix/poisson/)), making Poisson EPCA particularly suitable for compressing discrete distribution data. This capability is crucial for applications like belief compression in reinforcement learning [@Roy], where high-dimensional belief states can be effectively reduced with minimal information loss.

<!-- todo: smooth out transition -->

One notable example is in reinforcement learning, specifically in belief state compression for partially observable Markov decision processes (POMDPs). Using Poisson EPCA, the package effectively reduces high-dimensional belief spaces with minimal information loss, as demonstrated by recreating results from @shortRoy. In this case, Poisson EPCA achieved a nearly perfect reconstruction of a $41$-dimensional belief profile using just five basis components [CITE `CompressedBeleifMDPS.jl`, PAPER IN PRE-REVIEW]. Poisson EPCA compression also produces a more interpretable compression than traditional PCA because it minimizes the generalized KL divergence rather than the Frobenius norm.

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