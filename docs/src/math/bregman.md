# Bregman Divergences

Bregman divergences play a central role in understanding the probabilistic framework of EPCA. At a high level, Bregman divergences generalize the concept of distance between two points, though they do not necessarily satisfy the properties of a traditional metric (such as symmetry or the triangle inequality). Instead, Bregman divergences provide a flexible way to measure differences between data points, making them useful for applications in clustering, optimization, and information theory.

## Definition

Formally, the Bregman divergence [Bregman](@cite) $B_F$ associated with a function $F(\theta)$ is defined as

```math
B_F(p, q) = F(p) - F(q) - \langle f(p), p - q \rangle
```

where $F$ is a strictly convex and continuously differentiable function, $f = \nabla_x F$ is the convex conjugate (defined later) of $F$, and $\langle , \rangle$ denotes an inner product.Intuitively, the Bregman divergence expresses the difference at $p$ between $F$ and its first-order Taylor expansion about $q$.

## The Exponential Family

A distribution is said to be in the natural *exponential family* if its density can be written

```math
p(x ; \theta) = P_0(x) \exp(\langle x, \theta \rangle - G(\theta) )
```

where ``x`` and ``\theta`` are vectors in ``\mathbb{R}^d``, ``P_0`` is a known function that does not depend on ``\theta``, and ``G`` is the log-partition function.  Intuitively, the log-partition function ensures that the ``p`` is a valid distribution, meaning it integrates to ``1``

```math
G(\theta) = \log \int P_0(x) \exp(\langle x, \theta \rangle) dx.
```

We call ``\theta`` the *natural parameter* and ``\mu = \mathbb{E}_{\theta \sim p(\cdot; \theta)}[x]`` the *expectation parameter*. For the exponential family (and assuming some standard regularity conditions), we have ``\mu = \nabla_\theta G(\theta) \equiv g(\theta)`` [GLM, azoury](@cite). Since ``G`` is strictly convex, we can also define the inverse ``g^{-1}(\mu) \equiv \theta``.

## The Legendre Transform

To understand the relationship between expectation parameters and natural parameters, first recall the Legendre transform from physics. For a convex function ``h``, the Legendre transform is 

```math
h^*(\tilde{x}) \equiv \tilde{x} \cdot x - f(x).
```

We say that ``h^*`` is the *dual* (or convex conjugate) of ``h``. Let ``F`` be the dual ``G``

```math
F(\mu) \equiv \langle \mu, \theta \rangle - G(\theta).
```

Observe that the gradient of the dual is the inverse of the gradient of the log-partition,

```math
\begin{aligned}
f(\mu) 
&\equiv \nabla_\mu F(\mu) \\
&= \nabla_\mu \Big[ \langle \mu, \theta \rangle - G(\theta)\Big] \\
&= \theta + \langle \mu, \nabla_\mu \theta \rangle - \langle g(\theta), \nabla_\mu \theta \rangle \\
&= \theta \\
&= g^{-1}(\mu).
\end{aligned}
```

In summary, the parameterizations are related by the Legendre transformations 

```math
\nabla_\theta G(\theta) = g(\theta) = \mu
```

and

```math
\nabla_\mu F_(\mu) = f(\mu) = \theta.
```

## Bregman Loss Functions

The key relationship between members of the exponential family and the Bregman divergence is this: minimizing the negative log-likelihood of ``p(x, \theta)`` is equivalent to minimizing the Bregman divergence ``B_F``. To see this, first recall that the negative log-likelihood for members of the exponential family is

```math
\langle x, \theta \rangle - G(\theta).
```

## Other Resources

- Mark Reid provides an excellent and accessible [introduction](https://mark.reid.name/blog/meet-the-bregman-divergences.html) to Bregman divergences.
