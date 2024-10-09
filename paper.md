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

Principal component analysis (PCA) [@PCA1; @PCA2; @PCA3] is popular for compressing, denoising, and interpreting high-dimensional data, but it can underperform on binary, count, and composition data because the model assumes data is normally distributed. Exponential family PCA (EPCA) [@EPCA] extends PCA to handle any exponential family distribution, making it more suitable for fields where these data types are common, such as geochemistry, marketing, genomics, political science, and machine learning [@composition; @elements].

`ExpFamilyPCA.jl` is a library for EPCA written in Julia, a dynamic language for scientific computing [@Julia]. It is the first Julia package to implement EPCA and the first in any language to support multiple distributions for EPCA.

# Statement of Need

EPCA has been applied in reinforcement learning [@Roy] and more recently in sample debiasing [@debiasing] and finance [@finance]. Wider adoption, however, remains limited due to the lack of implementations, with the only existing package supporting a single distribution in MATLAB [@epca-MATLAB]. This is surprising, as other Bregman-based optimization techniques have been successful in areas like mass spectrometry [@spectrum], ultrasound denoising [@ultrasound], topological data analysis [@topological], and robust clustering [@clustering]. These successes indicate that EPCA holds untapped potential in signal processing and machine learning.

The lack of a general EPCA library is likely due to engineering challenges. In widely used languages like Python and C, fast symbolic differentiation and optimization libraries are not typically interoperable. Julia, by contrast, uses multiple dispatch which promotes high levels of generic code reuse [@dispatch]. Multiple dispatch allows `ExpFamilyPCA.jl` to integrate fast symbolic differentiation [@symbolics], optimization [@optim], and numerically stable computation [@stable_exp] without requiring costly API conversions. As a result, `ExpFamilyPCA.jl` delivers speed, stability, and flexibility, with built-in support for most common distributions (§ [Supported Distributions](#supported-distributions)) and flexible constructors for custom distributions (§ [Custom Distributions](#supported-distributions)).

## Principal Component Analysis

### Geometric Interpretation

Given a data matrix $X \in \mathbb{R}^{n \times d}$ with $n$ observations and $d$ features, PCA seeks the closest low-rank approximation $\Theta \in \mathbb{R}^{n \times d}$ by minimizing the reconstruction error

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \frac{1}{2}\|X - \Theta\|_F^2 \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}$$

where $\| \cdot \|_F$ denotes the Frobenius norm. The optimal $\Theta$ is a rank-$k$ hyperplane that can be decomposed as a matrix product

$$
X \approx \Theta = AV,
$$

where $A \in \mathbb{R}^{n \times k}$ is the score matrix and $V \in \mathbb{R}^{k \times d}$ is the matrix of the $k$-largest principal components (the eigenvectors of the covariance matrix). This suggests that each observation (a row of $X$) can be well-approximated by a linear combination of $k$ basis vectors (the rows of $V$):

$$
x_i \approx \theta_i = a_i V
$$

for $i = 1, \dots, n$.

### Probabilistic Interpretation

The PCA objective is equivalent to maximum likelihood estimation for a Gaussian model. Under this lens, PCA views the rows of $X$ as realizations of a $d$-dimensional Gaussian. Alternatively, each observation $x_i$ is a noisy observation of a latent, low-rank structure $\theta_i$ corrupted with high-dimensional Gaussian noise

$$
x_i \sim \mathcal{N}(\theta_i, I).
$$

To recover the structure $\Theta$, PCA solves

$$\begin{aligned}
& \underset{\Theta}{\text{maximize}}
& & \sum_{i=1}^{n}\log \mathcal{L}(x_i; \theta_i) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}$$

where $\mathcal{L}$ is the likelihood function.

## Exponential Family PCA

### Bregman Divergences

Bregman divergences [@Bregman; @Brad] are a measure of statistical difference that we can use to generalize the probablistic PCA objective to the exponential family. The Bregman divergence $B_F$ for a strictly convex, continuously differentiable function $F$ is

$$
B_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle.
$$

This can be interpreted as the difference between $F(p)$ and its linear approximation about $q$. When $F$ is taken to be the convex conjugate of the log-partition of an exponential family distribution, minimizing the Bregman divergence is the same as maximizing the corresponding log-likelihood [@azoury; @forster] (see [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/bregman/)). Since the Gaussian distribution is in the exponential family, this means that Bregman divergences generalize the PCA objective.