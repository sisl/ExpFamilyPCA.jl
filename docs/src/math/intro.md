# Introduction

Welcome to the first of three pages exploring exponential family principal component analysis (EPCA) [EPCA](@cite). EPCA is a robust dimensionality reduction technique that extends traditional principal component analysis (PCA) [PCA](@cite) by providing a probabilistic framework capable of handling complex, high-dimensional data with greater flexibility. Whether you are a data scientist, machine learning researcher, or mathematician, this guide will offer an accessible yet rigorous look into EPCA, building on foundational knowledge of multivariable calculus and linear algebra.

In this introduction, we’ll provide an overview of the basics and motivation behind EPCA. While advanced mathematical concepts will be covered, I aim to present them in a way that's approachable, even for readers less familiar with the subtleties of these topics. For a more concise or formal treatment, refer to the [original EPCA paper](@cite EPCA).

This guide is organized into several pages, each building on the last:

1. Introduction to EPCA (this page)
2. [Bregman Divergences](./bregman.md)
3. [EPCA Objectives and Derivations](./objectives.md)
4. Appendix
   - [The Gamma EPCA Objective is the Itakura-Saito Distance](./appendix/gamma.md)
   - [The Poisson EPCA Objective is the Generalized KL-Divergence](./appendix/poisson.md)
   - [The Inverse of the Link Function is the Gradient of the Convex Conjugate](./appendix/inverses.md)

5. [References](./references.md)

## Principal Component Analysis (PCA)

PCA is one of the most widely used dimension reduction techniques, particularly in data science and machine learning. It transforms high-dimensional data into a lower-dimensional space while retaining as much of the data’s variability as possible. By identifying directions of maximum variance—known as principal components—PCA provides an efficient way to project the data onto these new bases, making it easier to visualize, analyze, and interpret.

Applications of PCA are broad, from noise reduction and feature extraction to exploratory data analysis and visualization of complex datasets. In this section, we will focus on two primary interpretations of PCA: a geometric view and a probabilistic approach that links PCA with Gaussian noise reduction.

### Geometric Interpretation

The geometric interpretation of PCA is intuitive and grounded in linear algebra. Given a high-dimensional dataset $X \in \mathbb{R}^{n \times d}$, the goal is to find a lower-dimensional representation that best approximates the original data. PCA seeks the closest low-dimensional subspace by minimizing the distance between the data and its projection onto this subspace.

This can be formulated as an optimization problem where we find the rank-$k$ approximation $\Theta$ of the data matrix $X$ by minimizing the reconstruction error:

```math
\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \|X - \Theta\|_F \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k
\end{aligned}
```

where $\| \cdot \|_F$ denotes the Frobenius norm. The Frobenius norm is calculated as the square root of the sum of the squared differences between corresponding elements of the two matrices. Intuitively, it can be seen as an extension of the Euclidean distance for vectors, applied to matrices by flattening them into large vectors. This makes the Frobenius norm a natural way to measure how well the lower-dimensional representation approximates the original data.

To find the approximation, we decompose $\Theta$ into a product $\Theta = AV$ where $A \in \mathbb{R}^{n \times k}$ contains the projected data in the lower-dimensional space, and $V \in \mathbb{R}^{k \times d}$ is the matrix of principal component, which define the new orthogonal axes. 

### Probabilistic Interpretation

In addition to its geometric interpretation, PCA can also be understood from a probabilistic perspective, particularly when the data is assumed to be corrupted by Gaussian noise. From this viewpoint, PCA is not just about finding a low-dimensional subspace, but about recovering the most likely low-dimensional structure underlying noisy high-dimensional observations.

In the probabilistic formulation, we assume that each observed data point $x_i \in \mathbb{R}^{d}$ is a random sample from a Gaussian with mean $\theta_i \in \mathbb{R}^{k}$ and unit covariance: 

```math
x_i \sim \mathcal{N}(\theta_i, I).
```

The goal of PCA here is to find the parameters $\Theta = [\theta_1, \dots, \theta_n]$ that maximize the likelihood of observing the data under the Gaussian model. Maximizing the log-likelihood for this Gaussian model leads to the following expression:

```math
\ell(\Theta; X) = \frac{1}{2} \sum_{i=1}^{n} (x_i-\theta_i)^2
```

which is equivalent to minimizing the Frobenius norm in the geometric interpretation.

 ## Exponential Family PCA (EPCA)

Exponential Family Principal Component Analysis (EPCA) extends PCA by incorporating the flexibility of generalized linear models (GLMs) [GLM](@cite). Just as GLMs extend linear regression to handle a variety of response distributions, EPCA generalizes PCA to accommodate data with noise drawn from any exponential family distribution, rather than just Gaussian noise. This allows EPCA to address a broader range of real-world data scenarios where the Gaussian assumption may not hold.

At its core, EPCA replaces the geometric PCA objective with a more general probabilistic objective that minimizes the generalized Bregman divergence—a measure closely related to the exponential family—rather than the Frobenius norm, which PCA uses. This makes EPCA particularly versatile for dimensionality reduction when working with non-Gaussian data distributions.

```math
\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F(X \| g(\Theta)) + \epsilon B_F(\mu \| g(\Theta)) \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) = k.
\end{aligned}
```
In this formulation:
*  $g$ is the **link function** and the derivative of $G$. Since $G$ is strictly convex and continuously differentiable, $g$ is invertible.
*  $F$ is the **convex conjugate** or dual of $G$. A deeper discussion of duality and the Legendre transform is provided in the next section.
*  $B_F(p \| q)$ is the **Bregman divergence** induced from $F$.

On the [next page](./bregman.md), we dive deeper into Bregman divergences—a key concept underlying EPCA. We’ll explore their properties and how they connect to the exponential family, providing a solid foundation for understanding the probabilistic framework of EPCA. 