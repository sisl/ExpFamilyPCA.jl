# Pareto EPCA

## Math

| Name             | `ParetoEPCA`                      |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``-\log(-1 - \theta) + \theta \log m`` |
| ``g(\theta)``    | ``\log m - \frac{1}{\theta + 1}`` |
| ``\mu`` Space[^1]    | ``\mathbb{R} \setminus \{ \log{m} \}``                  |
| ``\Theta`` Space | negative                  |
| Appropriate Data | heavy-tail                        |
| ``m``            | ``m > 0`` (minimum value)                         |

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
ParetoEPCA
```

!!! tip
    If your compression converges to a constant matrix, try processing your data to reduce the maximum (e.g., divide your data by a large constant, take the logarithm).