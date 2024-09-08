# Weibull EPCA

| Name             | `WeibullEPCA`                     |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``G(\theta) = -\log(-\theta) - \log k``     |
| ``g(\theta)``    | ``-\frac{1}{\theta}``             |
| ``\mu`` Space    | ``\mathbb{R} / \{ 0 \}``                   |
| ``\Theta`` Space | ``(-\infty, 0)``                  |
| Appropriate Data | nonnegative continuous               |

`WeibullEPCA` omits it the known shape parameter ``k`` since it does not affect the Weibull EPCA objective.

## Documentation

```@docs
WeibullEPCA
```