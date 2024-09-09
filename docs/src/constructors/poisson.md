# Poisson EPCA

| Name             | `PoissonEPCA`                     |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``e^\theta``           |
| ``g(\theta)``    | ``e^\theta``                      |
| ``\mu`` Space[^1]    | positive                   |
| ``\Theta`` Space | real             |
| Appropriate Data | count, probability                             |

Poisson EPCA minimizes the generalized KL divergence making it well-suited for compressing probability profiles. Poisson EPCA has also been used in reinforcement learning to solve partially observed Markov decision processes (POMDPs) with belief compression [Roy](@cite). 

[^1]: ``\mu`` space refers to the space of valid *regularization parameters*, not to the *expectation parameter space*.

## Documentation

```@docs
PoissonEPCA
```