# Negative Binomial EPCA

| Name             | `NegativeBinomialEPCA`            |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``-r \log(1 - e^\theta)`` |
| ``g(\theta)``    | ``- \frac{r \exp \theta}{1 - \exp \theta}`` |
| ``\mu`` Space    | ``(0, \infty)``                   |
| ``\Theta`` Space | ``(-\infty, 0)``                  |
| Appropriate Data | count                             |
| ``r``            | ``r > 0`` (number of failures)    |


## Documentation

```@docs
NegativeBinomialEPCA
```