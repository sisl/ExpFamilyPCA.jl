# Negative Binomial EPCA

## Description

## Math

The cumulant of the negative binomial distribution with a known number of failures $r$ is

```math
G(\theta) = -r \log(1 - \exp \theta)
```

so the link function is

```math
g(\theta) = - \frac{r \exp \theta}{1 - \exp \theta}.
```

## Documentation

```@docs
NegativeBinomialEPCA
```