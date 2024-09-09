# Continuous Bernoulli EPCA

## Math

| Name             | `ContinuousBernoulliEPCA`                 |
|------------------|---------------------------------|
| ``G(\theta)``    | ``\log \frac{e^\theta - 1}{\theta}``         |
| ``g(\theta)``    | ``\frac{\theta-1}{\theta} + \frac{1}{e^\theta - 1}`` |
| ``\mu`` Space[^1]    | ``(0, 1) \setminus \{\frac{1}{2}\}``                      |
| ``\Theta`` Space | real              |
| Appropriate Data | unit interval                          |

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.


## Documentation

```@docs
ContinuousBernoulliEPCA
```
