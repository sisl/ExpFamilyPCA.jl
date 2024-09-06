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

## Documentation

```@docs
ParetoEPCA
```