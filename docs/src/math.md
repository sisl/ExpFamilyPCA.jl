# Math Details

## Principal Component Analysis (PCA)

PCA is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional subspace while retaining as much variability as possible. It accomplishes this by identifying the directions of maximum variance in the data, known as principal components, and projecting the data onto these new orthogonal axes. PCA finds extensive applications in various domains, including data visualization, noise reduction, feature extraction, and exploratory data analysis.

PCA can be formulated as an low-rank matrix approximation problem. For a data matrix $X$, we want to find low-dimensional approximation $\Theta$ that minimizes the the sum of the squared Euclidean distances. Formally, we write

\begin{mini}|l|
	  {\Theta}{\|X - \Theta\|_F}{}{}
	  \addConstraint{\mathrm{rank}\left(\Theta\right) \leq \ell}
 \end{mini}

 where $\| \cdot \|_F$ is the Frobenius norm. Observe that the objective is equivalent to maximizing the log likelihood of a Gaussian model. Consequently, PCA can be viewed as a denoising procedure that recovers the true low-dimensional signal $\Theta$ from a normally noised high-dimensional measurement $X$. 

 ## Exponential Family PCA (EPCA)


EPCA is an extension of PCA analogous to how generalized linear models \citep{McCullagh1989-js} extend linear regression. In particular, EPCA can denoise from any exponential family. \cite{collins2001generalization} showed that maximizing the log likelihood of any exponential family is directly related to minimizing the Bregman divergence

\begin{equation}
    B_F(p \| q) \equiv F(p) - F(q) - f(q)(p - q)
\end{equation}

where 

\begin{align}
    f(\mu) &\equiv \nabla_\mu F(\mu) \\
    F(\mu) &\equiv \theta \cdot g(\theta) - G(\theta)
\end{align}

and $g$ is the link function, $g(\theta) = \nabla_\theta G(\theta)$, and $\mu = g(\theta)$. In other words, $F$ is the convex dual of cumulant \citep{azoury2001relative}. We can now express the general formulation of the EPCA problem. For any differentiable convex function $G$, the EPCA problem is

 \label{prob:epca-objective}
\begin{mini}|l|
	  {\Theta}{B_F\left(X \| g(\Theta) \right)}{}{}
 \end{mini}

 <!-- % TODO: How to fix reference -->
 where $g$ is applied elementwise across $\Theta$ and $B_F$ is the generalized Bregman divergence. Unfortunately, the optimal convergence constraints of the general problem remain unsolved. As such, in practice we minimize a different objective

 \begin{mini}|l|
    {\Theta}{B_F\left(X \| g(\Theta) \right) + \epsilon B_F\left(\mu_0 \| g(\Theta)\right)}{}{}
 \end{mini}
 
 where $\mu_0$ in any value in $\mathrm{range}(g)$ and $\epsilon$ is some small positive constant.