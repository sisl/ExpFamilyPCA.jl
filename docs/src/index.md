# ExpFamilyPCA.jl Documentation

**ExpFamilyPCA.jl** is a Julia package for performing exponential principal component analysis (EPCA) [EPCA](@cite). EPCA generalizes PCA to accommodate data from any exponential family distribution, making it more suitable for fields where these data types are common, such as geochemistry, marketing, genomics, political science, and machine learning [composition, elements](@cite).

ExpFamilyPCA.jl the first EPCA package in Julia and the first in any language to support EPCA for multiple distributions and custom objectives.

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

## Supported Distributions

| Distribution         | `ExpFamilyPCA.jl`                 | Objective                                                         | Link Function `` g(\theta) ``                                    |
|----------------------|-----------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------|
| Bernoulli            | [`BernoulliEPCA`](./constructors/bernoulli.md)                   | `` \log(1 + e^{\theta - 2x\theta}) ``                             | `` \frac{e^\theta}{1 + e^\theta} ``                              |
| Binomial             | [`BinomialEPCA`](./constructors/binomial.md)                    | `` n \log(1 + e^\theta) - x\theta ``                              | `` \frac{ne^\theta}{1 + e^\theta} ``                             |
| Continuous Bernoulli | [`ContinuousBernoulliEPCA`](./constructors/continuous_bernoulli.md)         | `` \log\left(\frac{e^\theta - 1}{\theta}\right) - x\theta ``      | `` \frac{\theta - 1}{\theta} + \frac{1}{e^\theta - 1} ``         |
| Gamma¹    | [`GammaEPCA`](./constructors/gamma.md) or [`ItakuraSaitoEPCA`](./constructors/gamma.md) | `` -\log(-\theta) - x\theta ``                                   | `` -\frac{1}{\theta} ``                                          | 
| Gaussian² | [`GaussianEPCA`](./constructors/gaussian.md) or [`NormalEPCA`](./constructors/gaussian.md)    | `` \frac{1}{2}(x - \theta)^2 ``                                   | `` \theta ``                                                     |
| Negative Binomial    | [`NegativeBinomialEPCA`](./constructors/negative_binomial.md)            | `` -r \log(1 - e^\theta) - x\theta ``                             | `` \frac{-re^\theta}{e^\theta - 1} ``                            |
| Pareto               | [`ParetoEPCA`](./constructors/pareto.md)                      | `` -\log(-1 - \theta) + \theta \log m - x\theta ``                | `` \log m - \frac{1}{\theta + 1} ``                              |
| Poisson³  | [`PoissonEPCA`](./constructors/poisson.md)                     | `` e^\theta - x\theta ``                                          | `` e^\theta ``                                                   |
| Weibull              | [`WeibullEPCA`](./constructors/weibull.md)                     | `` -\log(-\theta) - x\theta ``                                    | `` -\frac{1}{\theta} ``                                          |

¹: For the Gamma distribution, the link function is typically based on the inverse relationship.  

²: For Gaussian, also known as Normal distribution, the link function is the identity. 
 
³: The Poisson distribution link function is exponential.

## Contributing

ExpFamilyPCA.jl was designed through the [Stanford Intelligent Systems Laboratory (SISL)](https://sisl.stanford.edu/) which helps maintain open-source projects posted on the [SISL organization repository](https://github.com/sisl) and the [JuliaPOMDPs](https://github.com/JuliaPOMDP) community. We subscribe to the community's [contribution guidelines](https://github.com/JuliaPOMDP/POMDPs.jl/blob/a14d1f3d2e1f551e154803064bc9496a0df4ba3e/CONTRIBUTING.md) and encourage new users with all levels of software development experience to contribute and ask questions. 

If you would like to create a new compressor or improve an existing model, please open a new issue that briefly describes the contribution you would like to make and the problem that it fixes, if there is one.
