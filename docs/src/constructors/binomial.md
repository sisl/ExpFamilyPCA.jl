# Binomial EPCA

## Math

| Name             | `BinomialEPCA`                 |
|------------------|---------------------------------|
| ``G(\theta)``    | ``n \log(1 + e^\theta)``         |
| ``g(\theta)``    | ``\frac{n e^\theta}{1+e^\theta}`` |
| ``\mu`` Space[^1]    | ``(0, n)``                      |
| ``\Theta`` Space | real         |
| Appropriate Data | count                          |
| ``n``            | ``n \geq 0`` (number of trials)                         |


``G`` is the scaled [softplus function](https://en.wikipedia.org/wiki/Softplus) and ``g`` is the scaled [logistic function](https://en.wikipedia.org/wiki/Logistic_function).

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.



## Documentation

```@docs
BinomialEPCA
```