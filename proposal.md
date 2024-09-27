# Summary

Principal component analysis (PCA) [@PCA] is a fundamental tool in data science and machine learning for dimensionality reduction and denoising. While PCA is effective for continuous, real-valued data, it may not perform well for binary, count, or discrete distribution data. Exponential family PCA (EPCA) [@EPCA] generalizes PCA to accommodate these data types, making it more suitable for tasks such as belief compression in reinforcement learning [@Roy]. `ExpFamilyPCA.jl` is the first Julia [@Julia] package for EPCA, offering fast implementations for common distributions and a flexible interface for custom distributions.

# Statment of Need

# Problem Formulation

## Principal Component Analysis

## The Exponential Family

## Bregman Divergences

## Exponential Family Principal Component Analysis

### Poisson

- math

- Example: Belief Compression

### Bernoulli

- math

- example for survey data w/ binary noise (e.g., yes no question set) w/ API usage

### Gamma

- math

- Example: Ultrasound Denoising
  TODO: follow the PCA denoising guide in the princeton tutorial