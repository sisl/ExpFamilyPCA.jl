# ExpFamilyPCA.jl

[![Build Status](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev-Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/)

**ExpFamilyPCA.jl** is a Julia package for performing [exponential principal component analysis (EPCA)](https://papers.nips.cc/paper_files/paper/2001/hash/f410588e48dc83f2822a880a68f78923-Abstract.html). ExpFamilyPCA.jl supports custom objectives and includes fast implementations for several common distributions.

## Documentation

For detailed documentation on each function and additional examples, please refer to the [documentation](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl).

## Installation

To install the package, use the Julia package manager. In the Julia REPL, type:

```julia
using Pkg; Pkg.add("ExpFamilyPCA")
```

## Quickstart

```julia
using ExpFamilyPCA

indim = 5
X = rand(1:100, (10, indim))  # data matrix to compress
outdim = 3  # target compression dimension

poisson_epca = PoissonEPCA(indim, outdim)

X_compressed = fit!(poisson_epca, X; maxiter=200, verbose=true)

Y = rand(1:100, (3, indim))  # test data
Y_compressed = compress(poisson_epca, Y; maxiter=200, verbose=true)

X_reconstructed = decompress(poisson_epca, X_compressed)
Y_reconstructed = decompress(poisson_epca, Y_compressed)
```

## Supported Models


| Distribution         | Objective                                              | Link Function $g(\theta)$                            |
|----------------------|--------------------------------------------------------|------------------------------------------------------|
| Bernoulli            | $\log(1 + e^{\theta-2x\theta})$                  | $\frac{e^\theta}{1+e^\theta}$                        |
| Binomial             | $n \log(1 + e^\theta) - x\theta$                   | $\frac{ne^\theta}{1+e^\theta}$                       |
| Continuous Bernoulli | $\log\Bigg(\frac{e^\theta -1}{\theta}\Bigg) - x\theta$ | $\frac{\theta - 1}{\theta} + \frac{1}{e^\theta - 1}$ |
| Gamma^1^               | $-\log(-x\theta) - x\theta$                            | $-1/\theta$                                          | 
| Gaussian^2^             | $\frac{1}{2}(x - \theta)^2$                            | $\theta$                                             |
| Negative Binomial    | $-r \log(1 - e^\theta) - x\theta$                  | $\frac{-re^\theta}{e^\theta - 1}$                    |
| Pareto               | $-\log(-1-\theta) + \theta \log m - x \theta$          | $\log m - \frac{1}{\theta+1}$                        |
| Poisson^3^              | $e^\theta - x \theta$                                  | $e^\theta$                                           |
| Weibull              | $-\log(-\theta) - x \theta$                            | $-1/\theta$                                          |

- ^1^: Equivalent to minimizing the [Itakura-Saito distance](https://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance)
- ^2^: Equivalent to vanilla PCA.
- ^3^: Equivalent to minimizing the generalized KL divergence.


## Custom Distributions

When working with custom distributions, it is often the case that certain specifications are more convenient than others. For example, writing the log-partition of the gamma distribution $G(\theta) = -\log(-\theta)$ and its derivative $g(\theta) = -1 / \theta$ is much simpler than coding the Itakura-Saito distance 

$$
\frac{1}{2\pi} \int_{-\pi}^{\pi} \Bigg[ \frac{P(\omega)}{\hat{P}(\omega)} - \log \frac{P(\omega)}{\hat{P}{\omega}} - 1\Bigg] d\omega
$$

effeciently in Julia even though the two are [equivalent](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/math/).

<!-- TODO: update the link to show the gamma math -->

ExpFamilyPCA.jl includes [X] constructors for custom distributions. All constrcutors are theoretically equivalent though some may be faster in practice. To showcase each constructor, we walk through how to construct a Poisson EPCA instance with each constructor. First, we provide a quick recap on notation.

<!-- TODO: double check if both G and F are strictly convesx -->

1. $G$ is the **log-partition function**. $G$ is strictly convex and continuously differentiable. 
2. $g$ is the **link function**. It is the derivative of the log-partition $\nabla_\theta G(\theta) = g(\theta)$ and the inverse of the derivative of the convex conjugate of the log-parition $g = f^{-1}$.
3. $F$ is the **convex conjugate** (under the [Legendre transform](https://en.wikipedia.org/wiki/Legendre_transformation)) of the log-partition $F = G^*$.
4. $f$ is the **derivative of the convex conjugate** $\nabla_x F(x) = f(x)$ and the inverse of the link function $f = g^{-1}$. 
5. $B_F(p \| q)$ is the [**Bregman divergence**](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/bregman/) induced from $F$.

For the Poisson distribution, these terms take the following values.

| Term        | Math                  | Julia                  |
|-------------|-----------------------|------------------------|
| $G(\theta)$ | $e^x$                 | `G = exp`               |
| $g(\theta)$ | $e^x$                 | `g = exp`               |
| $F(x)$      | $x \log x - x$        | `F(x) = x * log(x) - x`       |
| $f(x)$      | $\log x$              | `f(x) = log(x)`               |
| $B_F(p \| q)$ | $p \log(p/q) + q - p$ | `B(p, q) = p * log(p / q) + q - p` |
| $B_F(x \| g(\theta))$ | $e^\theta - x\theta + x \log x - x$ | `Bg(x, θ) = e^θ - x * θ + x * log(x) - x` |

The Bregman distance can also be specified using [Distances.jl](https://github.com/JuliaStats/Distances.jl)

```julia
using Distances

B = Distances.gkl_divergence
```

### Constructors

#### `EPCA1`

##### $F, g$

```julia
EPCA(indim, outdim, F, g, Val((:F, :g)))
```

##### $F, f$

```julia
EPCA(indim, outdim, F, f, Val((:F, :f)))
```

##### $F$

```julia
EPCA(indim, outdim, F, Val((:F)))
```

##### $F, G$

```julia
EPCA(indim, outdim, F, G, Val((:F, :G)))
```

#### `EPCA2`

##### $G, g$

```julia
EPCA(indim, outdim, G, g, Val((:G, :g)))
```

##### $G$

```julia
EPCA(indim, outdim, G, Val((:G)))
```

#### `EPCA3`

##### $B, G$

```julia
EPCA(indim, outdim, B, g, Val((:B, :g)))
```

##### $B, G$

```julia
EPCA(indim, outdim, B, G, Val((:B, :G)))
```

#### `EPCA4`

##### $Bg, g$ 

```julia
EPCA(indim, outdim, Bg, g, Val((:Bg, :g)))
```

##### $Bg, G$ 

```julia
EPCA(indim, outdim, Bg, G, Val((:Bg, :G)))
```

### Tips and Tricks

#### Metaprogramming

#### Dropping Constants

#### Selecting Constructors

#### Sobol Initialization

## Contributing

Contributions are welcome! If you want to contribute, please fork the repository, create a new branch, and submit a pull request. Before contributing, please make sure to update tests as appropriate.