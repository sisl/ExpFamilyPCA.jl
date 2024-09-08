# Bernoulli EPCA

| Name             | `BernoulliEPCA`                 |
|------------------|---------------------------------|
| ``G(\theta)``    | ``\log(1 + e^\theta)``         |
| ``g(\theta)``    | ``\frac{e^\theta}{1+e^\theta}`` |
| ``\mu`` Space    | ``(0, 1)``                      |
| ``\Theta`` Space | ``\mathbb{R}``                  |
| Appropriate Data | binary                          |

``G`` is the [softplus function](https://en.wikipedia.org/wiki/Softplus) and ``g`` is the [logistic function](https://en.wikipedia.org/wiki/Logistic_function).

## Documentation

```@docs
BernoulliEPCA
```
