# Gaussian EPCA

| Name             | `GaussianEPCA` or `NormalEPCA`    |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``\theta^2 / 2``     |
| ``g(\theta)``    | ``\theta``                        |
| ``\mu`` Space[^1]    | real             |
| ``\Theta`` Space | real                    |
| Appropriate Data | continuous                        |

The Gaussian EPCA objective is equivalent to regular PCA.

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
GaussianEPCA
```

```@docs
NormalEPCA
```