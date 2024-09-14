# The Link Function and the Expectation Parameter

Recall from the [page on Bregman divergences](../bregman.md) that the probability density function for a member of the natural exponential family is given by

```math
p_\theta(x) = h(x) \exp(x \theta - G(\theta))
```

where $G(\theta)$ is the log-partition function, defined as

```math
G(\theta) = \log \int h(x) \exp(x\theta) \, dx.
```

Now, by taking the gradient of the log-partition function $G(\theta)$, we get:

```math
\begin{aligned}
\nabla_\theta G(\theta) 
&= \nabla_\theta \left[ \log \int h(x) \exp(x \theta) \, dx \right] \\
&= \frac{ \int x \exp(x \theta) h(x) \, dx}{ \int \exp(x \theta) h(x) \, dx} \\
&= \frac{ \int x \exp(x \theta) h(x) \, dx}{ \exp(G(\theta))} \\
&= \int x \exp(x \theta - G(\theta)) h(x) \, dx \\
&= \int x p_\theta(x) \, dx \\
&= \mathbb{E}_\theta[X],
\end{aligned}
```

which is the desired result.
