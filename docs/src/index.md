# ExpFamilyPCA.jl Documentation

**ExpFamilyPCA.jl** is a Julia package for performing [exponential principal component analysis (EPCA)](https://papers.nips.cc/paper_files/paper/2001/hash/f410588e48dc83f2822a880a68f78923-Abstract.html). ExpFamilyPCA.jl supports custom objectives and includes fast implementations for several common distributions.

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
| Bernoulli            | `BernoulliEPCA`                   | `` \log(1 + e^{\theta - 2x\theta}) ``                             | `` \frac{e^\theta}{1 + e^\theta} ``                              |
| Binomial             | `BinomialEPCA`                    | `` n \log(1 + e^\theta) - x\theta ``                              | `` \frac{ne^\theta}{1 + e^\theta} ``                             |
| Continuous Bernoulli | `ContinuousBernoulliEPCA`         | `` \log\left(\frac{e^\theta - 1}{\theta}\right) - x\theta ``      | `` \frac{\theta - 1}{\theta} + \frac{1}{e^\theta - 1} ``         |
| Gamma¹    | `GammaEPCA` or `ItakuraSaitoEPCA` | `` -\log(-\theta) - x\theta ``                                   | `` -\frac{1}{\theta} ``                                          | 
| Gaussian² | `GaussianEPCA` or `NormalEPCA`    | `` \frac{1}{2}(x - \theta)^2 ``                                   | `` \theta ``                                                     |
| Negative Binomial    | `NegativeBinomialEPCA`            | `` -r \log(1 - e^\theta) - x\theta ``                             | `` \frac{-re^\theta}{e^\theta - 1} ``                            |
| Pareto               | `ParetoEPCA`                      | `` -\log(-1 - \theta) + \theta \log m - x\theta ``                | `` \log m - \frac{1}{\theta + 1} ``                              |
| Poisson³  | `PoissonEPCA`                     | `` e^\theta - x\theta ``                                          | `` e^\theta ``                                                   |
| Weibull              | `WeibullEPCA`                     | `` -\log(-\theta) - x\theta ``                                    | `` -\frac{1}{\theta} ``                                          |

¹: For the Gamma distribution, the link function is typically based on the inverse relationship.  

²: For Gaussian, also known as Normal distribution, the link function is the identity. 
 
³: The Poisson distribution link function is exponential.