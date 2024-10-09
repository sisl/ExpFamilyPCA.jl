# Weibull EPCA

| Name             | `WeibullEPCA`                     |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``-\log(-\theta) - \log k``     |
| ``g(\theta)``    | ``-\frac{1}{\theta}``             |
| ``\mu`` Space[^1]    | ``\mathbb{R} / \{ 0 \}``                   |
| ``\Theta`` Space | negative                  |
| Appropriate Data | nonnegative continuous               |

`WeibullEPCA` omits it the known shape parameter ``k`` since it does not affect the Weibull EPCA objective.

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
WeibullEPCA
```