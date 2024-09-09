# Negative Binomial EPCA

| Name             | `NegativeBinomialEPCA`            |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``-r \log(1 - e^\theta)`` |
| ``g(\theta)``    | ``- \frac{r  e^\theta}{1 - e^\theta}`` |
| ``\mu`` Space[^1]    | positive        |
| ``\Theta`` Space | negative                  |
| Appropriate Data | count                             |
| ``r``            | ``r > 0`` (number of successes)    |

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
NegativeBinomialEPCA
```