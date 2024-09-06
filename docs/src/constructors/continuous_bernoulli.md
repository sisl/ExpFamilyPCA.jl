# Continuous Bernoulli EPCA

## Description

## Math

The cumulant of the continuous Bernoulli distribution is

```math
G(\theta) = \log \frac{\exp \theta - 1}{\theta}
```

so the link function is

```math
g(\theta) = \frac{\theta-1}{\theta} + \frac{1}{\exp\theta - 1}.
```

## Documentation

```@docs
ContinuousBernoulliEPCA
```
