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

Dimensionality reduction techniques like principal component analysis (PCA) [@PCA] are fundamental tools in machine learning and data science for managing high-dimensional data. While PCA is effective for continuous, real-valued data, it may not perform well for binary, count, or discrete distribution data. Exponential family PCA (EPCA) [@EPCA] generalizes PCA to accommodate these data types, making it a more suitable choice for tasks like belief compression in reinforcement learning [@Roy]. `ExpFamilyPCA.jl` is the first Julia [@Julia] package for EPCA, offering fast implementations for common distributions and a flexible interface for custom objectives.

# Statement of Need

To our knowledge, there are no open-source implementations of EPCA and the sole proprietary package [@epca-MATLAB] is limited to a single distribution. Modern data science applications of EPCA in reinforcement learning [@Roy] and mass spectrometry [@spectrum] involve a diverse range of distributions and require numerical stability and the ability to handle large datasets. `ExpFamilyPCA.jl` addresses this gap by providing fast implementations for several exponential family distributions and multiple constructors for custom distributions. More implementation and mathematical details are in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/).

# Related Work

Exponential family PCA was introduced by [@EPCA], and several papers have extended the technique. While there have been advances, EPCA remains the most well-studied variation of PCA in the field of reinforcement learning and sequential decision-making [@Roy].

# Features

`ExpFamilyPCA.jl` includes efficient EPCA implementations for several exponential family distributions.

| Distribution             | Julia Type                  | Description                                            |
|--------------------------|-----------------------------|--------------------------------------------------------|
| Bernoulli                | `BernoulliEPCA`             | For binary data                                        |
| Binomial                 | `BinomialEPCA`              | For count data with a fixed number of trials           |
| Continuous Bernoulli     | `ContinuousBernoulliEPCA`   | For modeling probabilities between 0 and 1             |
| Gamma                    | `GammaEPCA`                 | For positive continuous data                           |
| Gaussian                 | `GaussianEPCA`              | Standard PCA for real-valued data                      |
| Negative Binomial        | `NegativeBinomialEPCA`      | For over-dispersed count data                          |
| Pareto                   | `ParetoEPCA`                | For modeling heavy-tailed distributions                |
| Poisson                  | `PoissonEPCA`               | For count and discrete distribution data               |
| Weibull                  | `WeibullEPCA`               | For modeling life data and survival analysis           |


# Acknowledgments

We thank Arec Jamgochian, Robert Moss, and Dylan Asmar for their help and guidance.

# References