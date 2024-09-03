# Exponential Family Distributions

## Preliminaries

Recall that the Bregman divergence is 

```math
B_F(p, q) = F(p) - F(q) - \langle f(p), p - q \rangle.
```

For simplicity, we drop the $\langle, \rangle$ notation. The (unregularized) EPCA objective seeks to minimize 

$$\begin{aligned}
B_F(X, g(\theta)) 
&= F(X) - F(g(\theta)) - f(g(\theta)) \cdot (X - g(\theta)) \\
&= F(X) - F(g(\theta)) - \theta \cdot (X - g(\theta))\\
&= F(X) - g(\theta) \theta + G(\theta) - \theta X + g(\theta)\theta\\
&= F(X) + G(\theta) - \theta X.
\end{aligned}$$
The second and third lines follow from the Legendre transform. Explicitly, the second line follows $f$ being the inverse of $g$, and the third lines follows from $F$ being the convex conjugate of $G$. Since $F(X)$ is a constant, we can drop the first term and write the final EPCA objective as 

$$
G(\theta) - \theta X.
$$

## Binomial

The cumulant of the binomial distribution with a known number of trials $n$ is

$$
G(\theta) = n \log(1 + \exp \theta)
$$

so the link function is

$$
g(\theta) = n \sigma(\theta)
$$

where $\sigma$ is the sigmoid.


## Negative Binomial

The cumulant of the negative binomial distribution with a known number of failures $r$ is

$$
G(\theta) = -r \log(1 - \exp \theta)
$$

so the link function is

$$
g(\theta) = - \frac{r \exp \theta}{1 - \exp \theta}.
$$


## Pareto

The cumulant of the Pareto distribution with a known minimum value $m$ is

$$
G(\theta) = -\log (-1 - \theta) + (1 + \theta) \log m
$$

so the link function is

$$
g(\theta) = \log m - \frac{1}{\theta + 1}.
$$

## Weibull

The cumulant of the Weibull distribution with a known shape $k$ is

$$
G(\theta) = -\log(-\theta) - \log k
$$

so the link function is

$$
g(\theta) = - \frac{1}{\theta}.
$$

## Continuous Bernoulli

The cumulant of the continuous Bernoulli distribution is

$$
G(\theta) = \log \frac{\exp \theta - 1}{\theta}
$$

so the link function is

$$
g(\theta) = \frac{\theta-1}{\theta} + \frac{1}{\exp\theta - 1}
$$