# The Inverse of the Link Function is the Gradient of the Convex Conjugate

Observe that the gradient of the dual is the inverse of the gradient of the log-partition,

```math
\begin{aligned}
f(g(\theta)) 
&= f(\mu) \\
&= \nabla_\mu F(\mu) \\
&= \nabla_\mu \Big[ \mu \theta - G(\theta)\Big] \\
&= \theta + \mu \nabla_\mu \theta - g(\theta) \nabla_\mu \theta \\
&= \theta. \\
\end{aligned}
```

The converse is similar, so $f = g^{-1}$.