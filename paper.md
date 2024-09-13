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
    orchid: 0000-0002-0164-3142
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 9 September 2024
bibliography: paper.bib
---

# Summary

Dimensionality reduction techniques like principal component analysis (PCA) [@PCA] are fundamental tools in machine learning and data science for managing high-dimensional data. While PCA is effective for continuous, real-valued data, it may not perform well for binary, count, or discrete distribution data. Exponential family PCA (EPCA) [@EPCA] generalizes PCA to accodate these data types, making it more a suitable choice for tasks like belief compression in reinforcement learning [@Roy]. `ExpFamilyPCA.jl` is the first Julia [@Julia] package for EPCA, offering fast implementations for common distributions and a flexible interface for custom objectives.

# Statement of Need

While PCA is widely available in various machine learning libraries, implementations of EPCA are more limited. Current EPCA solutions are mostly restricted to MATLAB [@epca-MATLAB], which is not always accessible or flexible for users working in open-source environments. Moreover, existing implementations often lack support for the diverse range of distributions found in modern data science applications, such as those seen in reinforcement learning [@Roy] and mass spectrometry [@spectrum].

By comparison, there has been no comprehensive open-source implementation of EPCA in Julia, a language increasingly used for numerical computing and data science. `ExpFamilyPCA.jl` fills this gap by providing a native Julia package for performing EPCA, supporting a wide range of exponential family distributions and a flexible interface for custom distributions.  This package offers improved numerical stability and efficiency, making it easier to handle large datasets. Furthermore, it introduces an accessible and high-performance tool for belief compression and other applications where EPCA can be particularly useful.

# Related Work

Exponential family PCA was introduced by @EPCA and several papers have extended the technique. @LitReview provide a comprehensive review of exponential PCA and its evolution. Although later authors have extended EPCA, exponential family PCA remains the most well-studied variation of PCA in the field of reinforcement learning and sequential decision making [@Roy]. To our knowledge the only implementation of exponential family PCA is in MATLAB [@epca-MATLAB].

## Exponential Family PCA

PCA can be interpreted as a Gaussian denoising procedure (see discussion in the [documentation](http://localhost:8000/math/intro/#Probabilistic-Interpretation)). EPCA extends this concept by generalizing PCA to handle noise drawn from *any* exponential family distribution.[^1] 

Before describing the EPCA objective, we introduce the necessary notation:


1. $G$ is the **log-partition function** of some exponential family distribution.
2. $g$ is the **link function** and the derivative of $G$. Since $G$ is strictly convex and continuously differentiable, $g$ is invertible.
3. $F$ is the **convex conjugate** or dual of $G$. A deeper discussion of duality and the Legendre transform is provided in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/bregman/#The-Legendre-Transform-and-Parameter-Duality).
4. $f$ is the derivative of $F$. Since $F$ is the convex conjugate of $G$, its derivative is the inverse link function $f = g^{-1}$ (see [])
5. $B_F(p \| q)$ is the [**Bregman divergence**](https://sisl.github.io/ExpFamilyPCA.jl/dev/bregman/) induced from $F$.


The EPCA objective is then written

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F(X \| g(\Theta)) + \epsilon B_F(\mu \| g(\Theta)) \\
& \text{subject to}
& & \mathrm{Rank}\left(\Theta\right) \leq \ell
\end{aligned}$$

where $\Theta$ is the natural parameter matrix and both $\epsilon > 0$ and $\mu \in \mathrm{Range}(g)$ are regularization hyperparameters that ensure the optimum is finite. See the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/epca/) for a deeper discussion of the EPCA objective.

[^1]: More generally, the EPCA objective can be induced from any contiuously-differentiable, strictly convex function.

## The `EPCA` Interface

The core of the `ExpFamilyPCA.jl` API is the `EPCA` abstract type. All supported and custom distributions are subtypes of `EPCA` and support the three methods in the `EPCA` interface: `fit!`, `compress` and `decompress`.

```julia
using ExpFamilyPCA

indim = 5
X = rand(1:100, (10, indim))  # data matrix to compress
outdim = 3  # target compression dimension

poisson_epca = PoissonEPCA(indim, outdim)

X_compressed = fit!(poisson_epca, X; maxiter=200, verbose=true)

Y = rand(1:100, (3, indim))  # test data
Y_compressed = compress(poisson_epca, Y; maxiter=200, verbose=true)

X_reconstructed = decompress(poisson_epca, X_compressed)
Y_reconstructed = decompress(poisson_epca, Y_compressed)
```

More details can be found in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/api/).

## Features

ExpFamilyPCA.jl includes fast EPCA implementations for 9 exponential family distributions and a collection of constructors for custom distributions.

### Supported Distributions

| Distribution         | `ExpFamilyPCA.jl`                 |
|----------------------|-----------------------------------|
| Bernoulli            | `BernoulliEPCA`                   |
| Binomial             | `BinomialEPCA`                    |
| Continuous Bernoulli | `ContinuousBernoulliEPCA`         |
| Gamma                | `GammaEPCA` or `ItakuraSaitoEPCA` |
| Gaussian             | `GaussianEPCA` or `NormalEPCA`    |
| Negative Binomial    | `NegativeBinomialEPCA`            |
| Pareto               | `ParetoEPCA`                      |
| Poisson              | `PoissonEPCA`                     |
| Weibull              | `WeibullEPCA`                     |

### Custom Distributions

When working with custom distributions, it is often the case that certain specifications are more convenient than others. For example, writing the log-partition of the gamma distribution $G(\theta) = -\log(-\theta)$ and its derivative $g(\theta) = -1 / \theta$ is much simpler than coding the Itakura-Saito distance [@ItakuraSaito]

$$
\frac{1}{2\pi} \int_{-\pi}^{\pi} \Bigg[ \frac{P(\omega)}{\hat{P}(\omega)} - \log \frac{P(\omega)}{\hat{P}{\omega}} - 1\Bigg] d\omega
$$

effeciently in Julia even though the two are equivalent (see [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/gamma/)).

There are many equivalent formulations of the EPCA objective and ExpFamilyPCA.jl supports many of them. Some constructors create objects that fit, decompress, and compress faster than others. To provide an example, we show how users can create `PoissonEPCA` using custom constructors. Each custom constructor requires some information about the desired distribution. The table below shows an example of that information for the Poisson distribution.

For the Poisson distribution, these terms take the following values.

| Term        | Math                  | Julia                  |
|-------------|-----------------------|------------------------|
| $G(\theta)$ | $e^x$                 | `G = exp`               |
| $g(\theta)$ | $e^x$                 | `g = exp`               |
| $F(x)$      | $x \log x - x$        | `F(x) = x * log(x) - x`       |
| $f(x)$      | $\log x$              | `f(x) = log(x)`               |
| $B_F(p \| q)$ | $p \log(p/q) + q - p$ | `B(p, q) = p * log(p / q) + q - p` |
| $B_F(x \| g(\theta))$ | $e^\theta - x\theta + x \log x - x$ | `Bg(x, θ) = e^θ - x * θ + x * log(x) - x` |

While users can simply use `poisson_epca = PoissonEPCA(indim, outdim)`, they could equivalently use any of the below constructors.

```julia
poisson_epca = EPCA(indim, outdim, F, g, Val((:F, :g)))
poisson_epca = EPCA(indim, outdim, F, f, Val((:F, :f)))
poisson_epca = EPCA(indim, outdim, F, Val((:F)))
poisson_epca = EPCA(indim, outdim, F, G, Val((:F, :G)))
poisson_epca = EPCA(indim, outdim, G, g, Val((:G, :g)))
poisson_epca = EPCA(indim, outdim, G, Val((:G)))
poisson_epca = EPCA(indim, outdim, B, g, Val((:B, :g)))
poisson_epca = EPCA(indim, outdim, B, G, Val((:B, :G)))
poisson_epca = EPCA(indim, outdim, Bg, g, Val((:Bg, :g)))
poisson_epca = EPCA(indim, outdim, Bg, G, Val((:Bg, :G)))
```

# TODO: include citation to CBMDP paper

# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References