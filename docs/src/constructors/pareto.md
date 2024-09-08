# Pareto EPCA

## Description

## Math

The cumulant of the Pareto distribution with a known minimum value $m$ is

```math
G(\theta) = -\log (-1 - \theta) + (1 + \theta) \log m
```

so the link function is

```math
g(\theta) = \log m - \frac{1}{\theta + 1}.
```

| Name             | `ParetoEPCA`                      |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``-\log(-1 - \theta) + \theta \log m`` |
| ``g(\theta)``    | ``\log m - \frac{1}{\theta + 1}`` |
| ``\mu`` Space    | ``(0, \infty)``                   |
| ``\Theta`` Space | ``(-1, \infty)``                  |
| Appropriate Data | continuous                        |
| ``m``            | ``m > 0`` (minimum value)                         |

## Documentation

```@docs
ParetoEPCA
```