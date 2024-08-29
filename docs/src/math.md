# Math Details

The goal of this page is to introduce and motivate exponential family principal component analysis (EPCA) [EPCA](@cite). This guide is accessible to anyone with knowledge of basic multivariable calculus and linear algebra (e.g., gradients, matrix rank).[^1] To ensure that readers can follow the math, we will write out every step and claim in exhaustive detail. We invite readers seeking a more concise, formal presentation of EPCA to explore the [original paper](@cite EPCA).

[^1]: If you are not yet familiar with these concepts, I suggest exploring these resources on [gradients](https://youtu.be/_-02ze7tf08?si=RzfLXbprHDQ-qSi-) and [matrix ranks](https://youtu.be/uQhTuRlWMxw?si=rH20Ih5A1mnyrR7f).

## Principal Component Analysis (PCA)

Principal component analysis [PCA](@cite) is an extremely popular dimensionality reduction technique. It has been invented by several researchers throughout its history and appears across many fields. There are many interpretations and derivations of PCA, but we will only discuss two here. The first is PCA as a low-rank matrix approximation problem; the second is as a Gaussian denoising problem. 

PCA [PCA](@cite) is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional subspace while retaining as much variability as possible. It accomplishes this by identifying the directions of maximum variance in the data, known as principal components, and projecting the data onto these new orthogonal axes. PCA finds extensive applications in various domains, including data visualization, noise reduction, feature extraction, and exploratory data analysis.

### Low-Rank Matrix Approximation

PCA can be formulated as an low-rank matrix approximation problem. For a data matrix $X$, we want to find low-dimensional approximation $\Theta$ that minimizes the the sum of the squared Euclidean distances. Formally, we write

$$\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & \|X - \Theta\|_F \\
& \text{subject to}
& & \mathrm{rank}\left(\Theta\right) \leq \ell
\end{aligned}$$

where $\| \cdot \|_F$ is the Frobenius norm. Observe that the objective is equivalent to maximizing the log-likelihood of a Gaussian model. Consequently, PCA can be viewed as a denoising procedure that recovers the true low-dimensional signal $\Theta$ from a normally noised high-dimensional measurement $X$. 

 ### Gaussian Denoising

 TODO: include image from presentation

 ## Exponential Family PCA (EPCA)

EPCA is an extension of PCA analogous to how generalized linear models [GLM](@cite) extend linear regression. In particular, EPCA can denoise from any exponential family. [EPCA](@citet) showed that maximizing the log-likelihood of any exponential family is directly related to minimizing the Bregman divergence

$$\begin{aligned} 
B_F(p \| q) \equiv F(p) - F(q) - f(q)(p - q) 
\end{aligned}$$

where 

$$\begin{aligned}
    f(\mu) &\equiv \nabla_\mu F(\mu) \\
    F(\mu) &\equiv \theta \cdot g(\theta) - G(\theta)
\end{aligned}$$

and $g$ is the link function, $g(\theta) = \nabla_\theta G(\theta)$, and $\mu = g(\theta)$. In other words, $F$ is the convex dual of cumulant \citep{azoury2001relative}. We can now express the general formulation of the EPCA problem. For any differentiable convex function $G$, the EPCA problem is

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

# References

```@bibliography
```