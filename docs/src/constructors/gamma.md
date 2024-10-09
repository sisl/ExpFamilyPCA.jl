# Gamma EPCA

## Math

| Name             | `GammaEPCA` or `ItakuraSaitoEPCA`                |
|------------------|---------------------------------|
| ``G(\theta)``    | ``-\log(-\theta)``         |
| ``g(\theta)``    | ``-\frac{1}{\theta}`` |
| ``\mu`` Space[^1]    | ``\mathbb{R} \setminus \{ 0 \}``                      |
| ``\Theta`` Space | negative                 |
| Appropriate Data | positive                         |

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
GammaEPCA
```

```@docs
ItakuraSaitoEPCA
```