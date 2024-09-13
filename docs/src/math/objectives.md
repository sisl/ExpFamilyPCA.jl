# Exponential Family Distributions

In this section, we describe several methods to induce the EPCA objective described [earlier](./intro.md):

```math
\begin{aligned}
& \underset{\Theta}{\text{minimize}}
& & B_F\left(X \| g(\Theta) \right) + \epsilon B_F\left(\mu \| g(\Theta)\right).
\end{aligned}
```

where ``B_F`` is the Bregman divergence associated with the convex conjugate ``F`` of the cumulant ``G`` of the distribution, ``X`` is our data matrix, and both ``\mu`` and ``\epsilon`` are hyperparameters used for normalization.

!!! note "Theorem 1"
    The EPCA objective and decompression can be expressed with ``F`` and ``g``.

## Case 1: ``F`` and ``g``

Recall that the Bregman divergence is 

```math
B_F(p, q) = F(p) - F(q) - \langle f(p), p - q \rangle.
```

For simplicity, we drop the ``\langle, \rangle`` notation. The (unregularized) EPCA objective seeks to minimize 

```math
\begin{aligned}
B_F(X, g(\theta)) 
&= F(X) - F(g(\theta)) - f(g(\theta)) \cdot (X - g(\theta)) \\
&= F(X) - F(g(\theta)) - \theta \cdot (X - g(\theta))
\end{aligned}
```

We can drop ``F(X)`` since it is a constant, so the EPCA objective can be written

```math
F(g(\theta)) - \theta \cdot (X - g(\theta))
```

meaning it can be fully specified in terms of ``F`` and ``g``. 

### ``F`` and ``f``

We can recover ``g`` using ``f``. To do so, recall that ``g = f^{-1}`` under the Legendre transform. Because the cumulant ``G`` is strictly convex, ``g`` is monotone increasing. Thus, we can evaluate ``g`` effeciently using a binary search. Explicitly, we can evaluate ``g`` at an arbitrary input ``a`` by finding the unique root of ``f(x) - a``. Thus, we can specify the EPCA objective and decompression with ``F`` and ``f``. 

### ``F``

We can differentiate ``F`` to find ``f`` and thus induce the EPCA objective.


## Case 2: ``G`` and ``g``

Recall that the Bregman divergence is 

```math
B_F(p, q) = F(p) - F(q) - \langle f(p), p - q \rangle.
```

For simplicity, we drop the ``\langle, \rangle`` notation. The (unregularized) EPCA objective seeks to minimize 

```math
\begin{aligned}
B_F(X, g(\theta)) 
&= F(X) - F(g(\theta)) - f(g(\theta)) \cdot (X - g(\theta)) \\
&= F(X) - F(g(\theta)) - \theta \cdot (X - g(\theta))\\
&= F(X) - g(\theta) \theta + G(\theta) - \theta X + g(\theta)\theta\\
&= F(X) + G(\theta) - \theta X.
\end{aligned}
```

The second and third lines follow from the Legendre transform. Explicitly, the second line follows ``f`` being the inverse of ``g``, and the third lines follows from ``F`` being the convex conjugate of ``G``. Since ``F(X)`` is a constant, we can drop the first term and write the final EPCA objective as 

```math
G(\theta) - \theta X.
```

This means that we can create an EPCA object by providing ``G`` and ``g`` (``g`` is needed for decompression since ``X_\text{recon} = g(AV)``), but we can find the link function ``g`` by symbolically differentiating ``G``, so EPCA can be entirely specified from the cumulant.

## Case 2: ``F`` and ``g``




