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

Principal component analysis (PCA) [@PCA] is a popular and well-studied technique for compression. One interpretation of PCA views it as a denoising procedure to recover the original low-dimensional projection from a high-dimensionally sample with Gaussian noise. Exponential family PCA (EPCA) [@EPCA] is an extension of PCA that accomodates noise drawn from any exponential family distribution. This approach is more appropriate for data that is not real-valued, such as binary and integer data. EPCA with Poisson loss can also effeciently represent probability distributions, making it useful for solving real-world sequential decision making problems through belief compression [@Roy]. Overall, this makes EPCA more effective for data representation and dimensionality reduction, particularly in machine learning and statistical modeling applications involving non-Gaussian data.

# Statement of Need

[ExpFamilyPCA.jl] is a Julia package for exponential family PCA as introduced in @EPCA. While there are many other extensions and varitaions of EPCA (TODO, cite many), we focus on @EPCA because of its direct relevance to belief compression [@Roy] in the field of sequential decision making.

TODO; discuss other related packages, also cite other JOSS submission

# Related Work

@LitReview provides a comprehensive review of exponential PCA and its evolution. We provide a summary here. 

## Exponential Family PCA

### PCA

PCA can be formuxlated as an low-rank matrix approximation problem. For a data matrix $X$, we want to find low-dimensional approximation $\Theta$ that minimizes the the sum of the squared Euclidean distances. Formally, we write

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \|X - \Theta\|_F \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) \leq \ell
\end{aligned}$$

where $\| \cdot \|_F$ is the Frobenius norm. Observe that the objective is equivalent to maximizing the log-likelihood of a Gaussian model. Consequently, PCA can be viewed as a denoising procedure that recovers the true low-dimensional signal $\Theta$ from a normally noised high-dimensional measurement $X$. 

### Formulation

EPCA is an extension of PCA analogous to how generalized linear models [@GLM] extend linear regression. In particular, EPCA can denoise from any exponential family. @Forester and @azoury showed that maximizing the log-likelihood of any exponential family is equivalent to minimizing the Bregman divergence

$$\begin{aligned} 
B_F(p \| q) \equiv F(p) - F(q) - \langle f(q), p - q \rangle
\end{aligned}$$

where 

$$\begin{aligned}
    f(\mu) &\equiv \nabla_\mu F(\mu) \\
    F(\mu) &\equiv \langle \theta , g(\theta) \rangle - G(\theta)
\end{aligned}$$

and $\langle \cdot, \cdot\rangle$ denotes an inner product, $g$ is the link function, $g(\theta) = \nabla_\theta G(\theta)$, and $\mu = g(\theta)$. In words, $F$ is the convex conjugate of the log-parition. We can now express the general formulation of the EPCA problem. For any differentiable convex function $G$, the EPCA problem is

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F\left(X \| g(\Theta) \right)
\end{aligned}$$

where $g$ is applied elementwise across $\Theta$ and $B_F$ is the generalized Bregman divergence. Unfortunately, the optimal convergence constraints of the general problem remain unsolved. As such, in practice we minimize a different objective

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F\left(X \| g(\Theta) \right) + \epsilon B_F\left(\mu_0 \| g(\Theta)\right)
\end{aligned}$$

where $\mu_0$ in any value in $\mathrm{range}(g)$ and $\epsilon$ is some small positive constant.

### Implementation

The primary object of ExpFamilyPCA.jl is the `EPCA` abstract type. There are many ways to construct an `EPCA` instance. Consider the Poisson EPCA. The cumulant of the Poisson is $G(\theta) = \exp \theta$, so the link function is $\nabla_\theta G(\theta) = \exp \theta$. The convex conjugate is $F(x) = x \log x - x$ and its gradient is $f(x) = \log x$ is the inverse link function as expected under the Legendre transform. 

```julia
using ExpFamilyPCA
using LogExpFunctions
using Distances

G = exp
g = exp
F(x) = x *logx(x) - x
f(x) = log(x + ϵ)
B = Distances.gkl_divergence
Bg(x, θ) = exp(θ) - x * θ + xlogx(x) - x
```

There are many ways to construct a Poisson EPCA.

```julia
epca(indim, outdim, F, g, Val((:F, :g)))
epca(indim, outdim, F, f, Val((:F, :f)))
epca(indim, outdim, F, Val((:F)))
epca(indim, outdim, F, G, Val((:F, :G)))
# TODO: add more
```

The derivation behind these equivalent constructions can be found in the documentation.

`EPCA` has several subclasses `EPCA1`, `EPCA2`, `EPCA3`, and `EPCA4`, none of which are exported

### Variants

ExpFamilyPCA.jl includes nine off-the-shelf `EPCA` subclasses: `BernoulliEPCA`, `BinomialEPCA`, `ContinuousBernoulliECPA`, `GammaEPCA` (`ItakuraSaitoEPCA`), `GaussianEPCA` (`NormalEPCA`), `NegativeBinomialEPCA`, `ParetoEPCA`, `PoissonEPCA`, and `WeibullEPCA`. Specific constructor details can be found in the [documentation](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/). 

## Usage



# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References