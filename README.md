# ExpFamilyPCA.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://sisl.github.io/ExpFamilyPCA.jl/stable)
[![Dev-Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://sisl.github.io/ExpFamilyPCA.jl/latest)
[![status](https://joss.theoj.org/papers/8c617a932d19b28d5ac0299b23d2c8dc/status.svg)](https://joss.theoj.org/papers/8c617a932d19b28d5ac0299b23d2c8dc)
[![Build Status](https://github.com/sisl/ExpFamilyPCA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sisl/ExpFamilyPCA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/sisl/ExpFamilyPCA.jl/branch/main/graph/badge.svg?token=kJESb0GybB)](https://codecov.io/github/sisl/ExpFamilyPCA.jl)

**ExpFamilyPCA.jl** is a Julia package for [exponential family principal component analysis (EPCA)](https://papers.nips.cc/paper_files/paper/2001/hash/f410588e48dc83f2822a880a68f78923-Abstract.html), a versatile generalization of PCA designed to handle non-Gaussian data, enabling dimensionality reduction and data analysis across a wide variety of distributions (e.g., binary, count, and compositional data). It is designed for applications in machine learning (belief compression, text analysis), signal processing (denoising), and data science (sample debiasing, clustering, dimensionality reduction), but can be applied to other fields with diverse data types.

- Website: https://sisl.github.io/ExpFamilyPCA.jl/dev/
- Math: https://sisl.github.io/ExpFamilyPCA.jl/dev/math/intro/
- API Documentation: https://sisl.github.io/ExpFamilyPCA.jl/dev/api/


## Features
- Implements exponential family PCA (EPCA)
- Supports multiple exponential family distributions
- Flexible constructors for custom distributions
- Fast symbolic differentiation and optimization
- Numerically stable scientific computation

## Installation

To install the package, use the Julia package manager. In the Julia REPL, type:

```julia
using Pkg; Pkg.add("ExpFamilyPCA")
```

## Supported Distributions
The following distributions are supported:

| Distribution                  | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `BernoulliEPCA`                | For binary data                                  |
| `BinomialEPCA`                 | For count data with a fixed number of trials     |
| `ContinuousBernoulliEPCA`      | For probabilities between 0 and 1                |
| `GammaEPCA`                    | For positive continuous data                     |
| `GaussianEPCA`                 | Standard PCA for real-valued data                |
| `NegativeBinomialEPCA`         | For over-dispersed count data                    |
| `ParetoEPCA`                   | For heavy-tailed distributions                   |
| `PoissonEPCA`                  | For count and discrete distribution data         |
| `WeibullEPCA`                  | For life data and survival analysis              |

## Quickstart
Each EPCA object supports the following methods:
- `fit!`: Trains the model and returns compressed training data.
- `compress`: Compresses new input data.
- `decompress`: Reconstructs original data from the compressed representation.

### Example:
```julia
X = sample_from_poisson(n1, indim)
Y = sample_from_poisson(n2, indim)
epca = PoissonEPCA(indim, outdim)

X_compressed = fit!(epca, X)
Y_compressed = compress(epca, Y)
Y_reconstructed = decompress(epca, Y_compressed)
```

The `sample_from_poisson` function is a placeholder for generating random Poisson-distributed data. It is not implemented in the code snippet to maintain clarity and focus on the core functionality of the example. If you wish to implement it, you can use the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package. For instance, you could define it as:

```julia
using Distributions

function sample_from_poisson(n::Int, dim::Int)
    d = Poisson()
    rand(d, n, dim)
end
```


## Custom Distributions

When working with custom distributions, certain specifications are often more convenient and computationally efficient than others. For example, inducing the gamma EPCA objective from the log-partition $G(\theta) = -\log(-\theta)$ and its derivative $g(\theta) = -1/\theta$ is much simpler than implementing the full the Itakura-Saito distance:

$$
D(P(\omega), \hat{P}(\omega)) =\frac{1}{2\pi} \int_{-\pi}^{\pi} \Bigg[ \frac{P(\omega)}{\hat{P}(\omega)} - \log \frac{P(\omega)}{\hat{P}{\omega}} - 1\Bigg] d\omega.
$$

In `ExpFamilyPCA.jl`, we would write:

```julia
G(θ) = -log(-θ)
g(θ) = -1 / θ
gamma_epca = EPCA(indim, outdim, G, g, Val((:G, :g)); options = NegativeDomain())
```

A lengthier discussion of the `EPCA` constructors and math is provided in the [documentation](https://sisl.github.io/ExpFamilyPCA.jl/dev/math/objectives/).

## Contributing

Contributions are welcome! If you want to contribute, please fork the repository, create a new branch, and submit a pull request. Before contributing, please make sure to update tests as appropriate.
