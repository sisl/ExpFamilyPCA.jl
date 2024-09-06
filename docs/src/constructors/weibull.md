# Weibull EPCA

## Description

## Math

The cumulant of the Weibull distribution with a known shape $k$ is

```math
G(\theta) = -\log(-\theta) - \log k
```

so the link function is

```math
g(\theta) = - \frac{1}{\theta}.
```

## Documentation

```@docs
WeibullEPCA
```