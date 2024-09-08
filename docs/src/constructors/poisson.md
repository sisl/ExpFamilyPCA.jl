# Poisson EPCA

| Name             | `PoissonEPCA`                     |
|------------------|-----------------------------------|
| ``G(\theta)``    | ``e^\theta``           |
| ``g(\theta)``    | ``e^\theta``                      |
| ``\mu`` Space    | ``(0, \infty)``                   |
| ``\Theta`` Space | ``\mathbb{R}``                    |
| Appropriate Data | count, probability                             |

Poisson EPCA minimizes the generalized KL divergence making it well-suited for compressing probability profiles. Poisson EPCA has also been used in reinforcement learning to solve partially observed Markov decision processes (POMDPs) with belief compression [Roy](@cite). 

## Documentation

```@docs
PoissonEPCA
```