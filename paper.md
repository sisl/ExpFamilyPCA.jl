---
title: 'ExpFamilyPCA.jl: A Julia Package for Exponential Family Principal Component Analysis'
tags:
  - POMDP
  - MDP
  - Julia
  - sequential decision making
  - RL
  - compression
  - dimensionality reduction
  - PCA
  - exponential family
  - E-PCA
  - open-source
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
date: 24 July 2024
bibliography: paper.bib
---

# Summary

Dimensionality reduction techniques like principal component analysis (PCA) [@PCA] are important for effeciently handling high-dimensional data in machine learning and data science. While PCA is appropriate for arbitrary real data, exponential family PCA (EPCA) [@EPCA] can be a better choice for compressing binary, integer, and probability data. EPCA with Poisson loss for example is useful for "belief compression" [@Roy] reinforcement learning and sequential decision making. 

# Statement of Need

[ExpFamilyPCA.jl](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl) is a Julia package for exponential PCA as introduced in @EPCA. 

Mention numerical stability. Unsure what to write here, since the summary already seems to cover the "need" of the package.

# Related Work

Exponential family PCA was introduced by @EPCA in 2001. Since then several papers have extended the technique. @LitReview provides a comprehensive review of exponential PCA and its evolution. Although later authors have extended EPCA, exponential family PCA remains the most well-studied variation in the field of reinforement learning and sequential decision making [@Roy]. To our knowledge the only implementation of exponential family PCA is in MATLAB [@epca-MATLAB].

## Exponential Family PCA

PCA can be viewed as a Gaussian denoising procedure (see discussion in the [documentation](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/math/epca/#The-Probabilistic-View)). EPCA extends the PCA formulation to accomodate noise drawn from *any* exponential family distribution.[^1] Explicitly, the EPCA objective is

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F(X \| g(\Theta)) + \epsilon B_F(\mu \| g(\Theta)) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) \leq \ell
\end{aligned}$$

where $g$ is the link function (the derivative of the log-partition $g(\theta) = \nabla_\theta G(\theta)$), $F$ is the convex conjugate of the log-partition, and both $\epsilon > 0$ and $\mu \in \text{Range(g)}$ are both hyperparameters used to regularize the objective (i.e., ensure real stationary points).

## Implementation

### Supported Distributions

| Distribution         | Objective                                              | Link Function $g(\theta)$                            |
|----------------------|--------------------------------------------------------|------------------------------------------------------|
| Bernoulli            | $\log(1 + e^{\theta-2x\theta})$                        | $\frac{e^\theta}{1+e^\theta}$                        |
| Binomial             | $n \log(1 + e^\theta) - x\theta$                       | $\frac{ne^\theta}{1+e^\theta}$                       |
| Continuous Bernoulli | $\log\Bigg(\frac{e^\theta -1}{\theta}\Bigg) - x\theta$ | $\frac{\theta - 1}{\theta} + \frac{1}{e^\theta - 1}$ |
| Gamma                | $-\log(-x\theta) - x\theta$                            | $-1/\theta$                                          | 
| Gaussian             | $\frac{1}{2}(x - \theta)^2$                            | $\theta$                                             |
| Negative Binomial    | $-r \log(1 - e^\theta) - x\theta$                      | $\frac{-re^\theta}{e^\theta - 1}$                    |
| Pareto               | $-\log(-1-\theta) + \theta \log m - x \theta$          | $\log m - \frac{1}{\theta+1}$                        |
| Poisson              | $e^\theta - x \theta$                                  | $e^\theta$                                           |
| Weibull              | $-\log(-\theta) - x \theta$                            | $-1/\theta$                                          |

The gamma EPCA objective is equivalent to minimizing the [Itakura-Saito distance](https://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance). The Gaussian EPCA objective is equivalent to usual PCA. The Poisson EPCA objective is equivalent to minimizing the generalized KL divergence.

### Custom Distributions

When working with custom distributions, it is often the case that certain specifications are more convenient than others. For example, writing the log-partition of the gamma distribution $G(\theta) = -\log(-\theta)$ and its derivative $g(\theta) = -1 / \theta$ is much simpler than coding the Itakura-Saito distance 

$$
\frac{1}{2\pi} \int_{-\pi}^{\pi} \Bigg[ \frac{P(\omega)}{\hat{P}(\omega)} - \log \frac{P(\omega)}{\hat{P}{\omega}} - 1\Bigg] d\omega
$$

effeciently in Julia even though the two are equivalent (see discussion in the [documentation](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/math/gamma/)).

ExpFamilyPCA.jl includes 10 constructors for custom distributions. All constrcutors are theoretically equivalent though some may be faster in practice. To showcase each constructor, we walk through how to construct a Poisson EPCA instance with each constructor. First, we provide a quick recap on notation (see discussion in the [documentation](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/math/bregman/)).

1. $G$ is the **log-partition function**. $G$ is strictly convex and continuously differentiable. 
2. $g$ is the **link function**. It is the derivative of the log-partition $\nabla_\theta G(\theta) = g(\theta)$ and the inverse of the derivative of the convex conjugate of the log-parition $g = f^{-1}$.
3. $F$ is the **convex conjugate** (under the [Legendre transform](https://en.wikipedia.org/wiki/Legendre_transformation)) of the log-partition $F = G^*$.
4. $f$ is the **derivative of the convex conjugate** $\nabla_x F(x) = f(x)$ and the inverse of the link function $f = g^{-1}$. 
5. $B_F(p \| q)$ is the [**Bregman divergence**](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/bregman/) induced from $F$.

For the Poisson distribution, these terms take the following values.

| Term        | Math                  | Julia                  |
|-------------|-----------------------|------------------------|
| $G(\theta)$ | $e^x$                 | `G = exp`               |
| $g(\theta)$ | $e^x$                 | `g = exp`               |
| $F(x)$      | $x \log x - x$        | `F(x) = x * log(x) - x`       |
| $f(x)$      | $\log x$              | `f(x) = log(x)`               |
| $B_F(p \| q)$ | $p \log(p/q) + q - p$ | `B(p, q) = p * log(p / q) + q - p` |
| $B_F(x \| g(\theta))$ | $e^\theta - x\theta + x \log x - x$ | `Bg(x, θ) = e^θ - x * θ + x * log(x) - x` |

The Bregman distance can also be specified using [Distances.jl](https://github.com/JuliaStats/Distances.jl)

```julia
using Distances

B = Distances.gkl_divergence
```

```julia
EPCA(indim, outdim, F, g, Val((:F, :g)))
EPCA(indim, outdim, F, f, Val((:F, :f)))
EPCA(indim, outdim, F, Val((:F)))
EPCA(indim, outdim, F, G, Val((:F, :G)))
EPCA(indim, outdim, G, g, Val((:G, :g)))
EPCA(indim, outdim, G, Val((:G)))
EPCA(indim, outdim, B, g, Val((:B, :g)))
EPCA(indim, outdim, B, G, Val((:B, :G)))
EPCA(indim, outdim, Bg, g, Val((:Bg, :g)))
EPCA(indim, outdim, Bg, G, Val((:Bg, :G)))
```

## Example Usage

Using ExpFamilyPCA.jl is simple. We provide a short example below.

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

## TODO: pottery demo
## TODO: connection w/ belief compression package

# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References