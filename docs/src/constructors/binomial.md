# Binomial EPCA

## Description

## Math

The cumulant of the binomial distribution with a known number of trials $n$ is

```math
G(\theta) = n \log(1 + \exp \theta)
```

so the link function is

```math
g(\theta) = n \sigma(\theta)
```

where $\sigma$ is the sigmoid.

## Documentation

```@docs
BinomialEPCA
```