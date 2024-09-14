# Bregman Divergences

Bregman divergences [Bregman](@cite) play a central role in the probabilistic framework of EPCA. They generalize the concept of distance between two points but do not necessarily satisfy the properties of a traditional metric (such as symmetry or the triangle inequality). Instead, Bregman divergences provide a flexible way to measure differences between data points, making them useful in applications like clustering, optimization, and information theory.

Understanding Bregman divergences is essential for EPCA because they link the exponential family of probability distributions to loss functions used in optimization, allowing us to generalize PCA to non-Gaussian data.

## Definition

Formally, the Bregman divergence [Bregman](@cite) $B_F$ associated with a function $F(\theta)$ is defined as

```math
B_F(p, q) = F(p) - F(q) - \langle f(p), p - q \rangle
```

where 
*  $F(\mu)$ is a strictly convex and continuously differentiable function, 
*  $f(\mu) = \nabla_\mu F(\mu)$ is the convex conjugate (defined later) of $F$, 
*  and $\langle \cdot, \cdot \rangle$ denotes an inner product.

Intuitively, the Bregman divergence expresses the difference at $p$ between $F$ and its first-order Taylor expansion about $q$.

### Properties

Unlike traditional metrics, Bregman divergences are not generally symmetric (i.e., $B_F(p \| q) \neq B_F(q \| p)$) and do not usually satsify the triangle inequality. However, they are always non-negative ($B_F(p \| q) \geq 0$) and equal $0$ if and only if $p = q$.

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

1.  **Natural Parameter** ($\theta$): This parameter controls the distribution’s shape in its canonical form. For example, the natural parameter for the Poisson distribution is $\log \lambda$.
2.  **Expectation Parameter** ($\mu$): This is the expected value of the sufficient statistic,[^1] computed as the mean of the data under the distribution. In exponential family distributions, it is related to the natural parameter through the gradient of the log-partition function:

```math
\mu = \mathbb{E}_{\theta}[X] = \nabla_\theta G(\theta) = g(\theta)
```
where $E_\theta$ is the expectation with respect to the distribution $p_\theta$. A derivation is provided in the [appendix](./appendix/expectation.md). Similarly, we also have $\theta = f(\mu)$.

[^1]: The sufficient statistic for the natural exponential family is simply the identity.

## The Legendre Transform

To understand the relationship between the natural parameters $\theta$ and the expectation parameters $\mu$, we introduce the concept of convex conjugates and the Legendre transform. For a convex function $F$, its convex conjugate (or dual) $F^*$ is defined as:[^2]

```math
F^*(\theta) = \sup_{\mu} [\langle \theta, \mu \rangle - F(\mu)].
```

The convex conjugate is an involution ($F^{**} = F$) meaning the Legendre transform allows us to convert back and forth between the natural and expectation parameter spaces. In the [next section](./objectives.md), we see how `ExpFamilyPCA.jl` exploits the rich mathematical structure of the Legendre transform to discover multiple specifications of the Bregman divergence.

[^2]: Duality also refers to the concept in convex analysis [convex](@cite).

## Bregman Loss Functions

An important relationship connects exponential family distributions and Bregman divergences: minimizing the negative log-likelihood of an exponential family distribution is equivalent (up to a constant) to minimizing a Bregman divergence between the observed data and the distribution's expectation parameter [azoury, forster](@cite). This connection is fundamental in extending PCA to EPCA.

Consider the negative log-likelihood of an exponential family distribution:

```math
-\ell(x; \theta) = G(\theta) - \langle x, \theta \rangle.
```

Our goal is to show that this expression is equivalent (up to a constant) to the Bregman divergence $B_F(x \| \mu)$. First, recall that $G$ is the convex conjugate (dual) of $F$, so:

```math
G(\theta) = \langle \theta, \mu \rangle - F(\mu).
```

Next, substitute $G$ back into the negative log-likelihood:

```math
\begin{aligned}
-\ell(x; \theta) &= (\langle \theta, \mu \rangle - F(\mu)) - \langle x, \theta \rangle \\
&= -F(\mu) - \langle \theta, x - \mu \rangle.
\end{aligned}
```
The last line is equivalent to $B_F(x \| \mu)$ up to a constant, meaning we can interpret the Bregman divergence as a loss function that generalizes the maximum likelihood of any exponential family distribution. In the [next section](./objectives.md), we expand this idea to derive specific constructors for the EPCA objective.
