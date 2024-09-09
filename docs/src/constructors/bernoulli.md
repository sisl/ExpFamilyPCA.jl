# Bernoulli EPCA

## Math

| Name             | `BernoulliEPCA`                 |
|------------------|---------------------------------|
| ``G(\theta)``    | ``\log(1 + e^\theta)``         |
| ``g(\theta)``    | ``\frac{e^\theta}{1+e^\theta}`` |
| ``\mu`` Space[^1]    | ``(0, 1)``                      |
| ``\Theta`` Space | real            |
| Appropriate Data | binary                          |

``G`` is the [softplus function](https://en.wikipedia.org/wiki/Softplus) and ``g`` is the [logistic function](https://en.wikipedia.org/wiki/Logistic_function).

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
BernoulliEPCA
```
