# Bregman Divergences

The EPCA objective is formulated as a Bregman divergence [Bregman](@cite). Bregman divergences are a measure of difference between two points (often probability distributions); however, they are not proper metrics, because they do not always satisfy symmetry and the triangle inequality.

## Definition

Formally, the Bregman divergence $B_F$ associated with a function $F(\theta)$ is defined as

```math
B_F(p \| q) = F(p) - F(q) - \langle f(p), p - q \rangle
```

where 
*  $F(\mu)$ is a strictly convex and continuously differentiable function, 
*  $f(\mu) = \nabla_\mu F(\mu)$ is the gradient of $F$, 
*  and $\langle \cdot, \cdot \rangle$ denotes an inner product.

Intuitively, the Bregman divergence expresses the difference at $p$ between $F$ and its first-order Taylor expansion about $q$.

### Aside: Properties

Bregman divergences vanish $B_F(p \| q) = 0$ if and only if their inputs also vanish $p = q = 0$. They are also always non-negative $B_F(p \| q) \geq 0$ for all $p, q \in \mathrm{domain}(F)$.

!!! info
    While the full EPCA objective is always non-negative, the `EPCA` loss may be negative because `ExpFamilyPCA.jl` uses transformed objectives that are equivalent but not equal to minimizing a sum of Bregman divergences.  

## The Exponential Family

The natural exponential family is a broad class of probability distributions that includes many common distributions such as the Gaussian, binomial, Poisson, and gamma distributions. A probability distribution belongs to the exponential family if its probability density function (or mass function for discrete variables) can be expressed in the following canonical form:

```math
p_\theta(x) = h(x) \exp(\langle \theta, x \rangle - G(\theta) )
```

where
*  $\theta$ is the **natural parameter** (also called the canonical parameter) of the distribution,
*  $h(x)$ is the base measure, independent of $\theta$,
*  and $G(\theta)$ is the **log-partition function** (also called the cumulant function) defined as:

```math
G(\theta) = \log \int h(x) \exp(\langle \theta, x \rangle) \, dx.
```

The log-partition function $G(\theta)$ ensures that the probability distribution integrates (or sums) to $1$.

### Key Parameters

1. The **natural parameter** $\theta$ controls the distribution’s shape in its canonical form. For example, the natural parameter for the Poisson distribution is $\log \lambda$.
2. The **expectation parameter** $\mu$ is the expected value of the sufficient statistic,[^1] computed as the mean of the data under the distribution. In exponential family distributions, it is related to the natural parameter through the **link function** $g$:

```math
\mu = \mathbb{E}_{\theta}[X] = \nabla_\theta G(\theta) = g(\theta)
```
where $E_\theta$ is the expectation with respect to the distribution $p_\theta$ (see [appendix](./appendix/expectation.md)). Similarly, we also have $\theta = f(\mu)$ (see [appendix](./appendix/inverses.md)).

[^1]: The sufficient statistic for the natural exponential family is simply the identity.

## Convex Conjugation

The fact that $f$ and $g$ are inverses follows from the stronger claim that $F$ and $G$ are convex conjugates. For a convex function $F$, its convex conjugate (or dual)[^2] $F^*$ is

```math
F^*(\theta) = \sup_{\mu} [\langle \theta, \mu \rangle - F(\mu)].
```

Convex conjugation is also an involution meaning it inverts itself, so $F^{**} = F$. Conjugation provides a rich structure for converting between natural and expectation parameters and, as we explain in the [next section](./objectives.md), helps induce multiple useful specifications of the EPCA objective.

[^2]: Duality also refers to the concept in convex analysis [convex](@cite).

## Bregman Loss Functions

Bregman divergences are crucial to EPCA, because they are equivalent (up to a constant) to maximum likelihood estimation for the exponential family [azoury, forster](@cite). To see this, consider the negative log-likelihood of such a distribution:

```math
-\ell(x; \theta) = G(\theta) - \langle x, \theta \rangle.
```

We want to show that this is equivalent to the Bregman divergence $B_F(x \| \mu)$. From the previous subsection, we know $G$ is the convex conjugate of $F$, so:

```math
G(\theta) = \langle \theta, \mu \rangle - F(\mu).
```

Substituting $G$ back into the negative log-likelihood, then yields:

```math
\begin{aligned}
-\ell(x; \theta) &= (\langle \theta, \mu \rangle - F(\mu)) - \langle x, \theta \rangle \\
&= -F(\mu) - \langle \theta, x - \mu \rangle.
\end{aligned}
```
The last line is equivalent to $B_F(x \| \mu)$ up to a constant, meaning we can interpret the Bregman divergence as a loss function that generalizes the maximum likelihood of any exponential family distribution. In the [next section](./objectives.md), we expand this idea to derive specific constructors for the EPCA objective.
