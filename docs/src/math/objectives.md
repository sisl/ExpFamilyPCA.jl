# Deriving the EPCA Objective Function

When working with custom distributions, certain specifications are often more convenient and computationally efficient than others. For example, expressing the log-partition function of the gamma distribution as $G(\theta) = -\log(-\theta)$ and its derivative $g(\theta) = -1/\theta$ is significantly simpler than implementing the Itakura-Saito distance [ItakuraSaito](@cite):

```math
D(P(\omega), \hat{P}(\omega)) = \frac{1}{2\pi} \int_{-\pi}^{\pi} \Bigg[ \frac{P(\omega)}{\hat{P}(\omega)} - \log \frac{P(\omega)}{\hat{P}{\omega}} - 1\Bigg] \, d\omega
```

even though both formulations are equivalent (see the [appendix](./appendix/gamma.md)). This example highlights the importance of flexibility in specifying mathematical components when the EPCA model. Choosing convenient representations is simpler and sometimes more effecient.

In this section, we demonstrate how the EPCA objective function and the decompression function $g$ can be derived using different combinations of mathematical components. This flexibility allows for efficient and adaptable implementations of EPCA in Julia.

## The Regularized EPCA Objective

Recall from the [introduction](./intro.md) that the regularized EPCA objective aims to minimize the following expression:

```math
B_{F}(X \| g(\Theta)) + \epsilon B_{F}(\mu_0 \| g(\Theta)).
```

where:

*  $B_F$ is the Bregman divergence generated by a convex and continuously differentiable function $F$,
*  $F$ is the convex conjugate of $G$,
*  $G$ is the log-parition function (or any convex function),
*  $g$ is the link function,
*  $X$ is the data matrix,
*  $\Theta$ is the parameter matrix, and
*  $\mu_0 \in \mathrm{range}(g)$ and $\epsilon > 0$ are regularization parameters.

Our goal is to show that both $B_F$ and $g$ can be induced from various base components, namely $F$, $G$, $f$ and $g$. This allows for multiple pathways to define and compute the EPCA objective in Julia.

## Different Approaches to Specifying the EPCA Objective and Decompression

### 1. Using $F$ and $g$

We begin by showing that the convex function $F$ and the link function $g$ are sufficient to define the EPCA objective. The Bregman divergence $B_F(X \| g(\Theta))$ can be expressed as

```math
\begin{aligned}
B_F(X \| g(\Theta)) &= F(X) - F(g(\Theta)) - f(g(\Theta))(X - g(\Theta)) \\
&= F(X) - F(g(\Theta)) - \Theta(X - g(\Theta))
\end{aligned}
```

where the second line follows from the relationship $f = g^{-1}$ (see the [appendix](./appendix/inverses.md)) and $\Theta = f(g(\Theta))$. Thus, specifying $F$ and $g$ is sufficient to fully describe the EPCA objective and decompression.

**Example:**
```julia
F(x) = x * log(x) - x
g(θ) = exp(θ)
poisson_epca = EPCA(indim, outdim, F, g, Val((:F, :g)))
```

### 2. Using $F$ and $f$

Since $g$ can be recovered from $f$ (the inverse of $g$), we can also derive the EPCA objective using $F$ and $f$. Given that $g$ is strictly increasing (as $G$, the conjugate of $F$, is strictly convex and differentiable), we can compute $g$ numerically.

To evaluate $g(a)$ for any $a$ in the domain of $g$, we solve for $x$ in the equation $f(x) = a$. Since $f$ and $g$ are monotone, we can effeciently solve for $x$ by finding the unique root of $f(a) - x$ using a binary search. Therefore, the EPCA objective can also be specified using only $F$ and $f$.

**Example:**
```julia
F(x) = x * log(x) - x
f(x) = log(x + eps())
poisson_epca = EPCA(indim, outdim, F, f, Val((:F, :f)))
```

### 3. Using $F$ Alone

`ExpFamilyPCA.jl`'s versatility is a direct result of two properties of Julia. The first is multiple dispatch. Julia's multiple dispatch system promotes high levels of generic code reuse [dispatch](@cite) meaning libraries used for symbolic differentiation like [`Symbolics.jl`](https://symbolics.juliasymbolics.org/stable/) [symbolics](@cite) can return base Julia atoms that work well with optimization libraries like [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/) [optim](@cite).

To induce the EPCA objective from $F$ alone, we first use `Symbolics.jl` to recover $f$. We convert $f$ to base Julia's second useful property: metaprogramming (non-metaprogramming conversion is available, though it generally performs much slower). Since both $F$ and $f$ are represented using generic Julia code, they seamlessly integrate with `Optim.jl`.

To recover $g$ for decompression, we can use the same procedure as described above (which is again possible because of multiple dispatch). Therefore, by defining $F$ alone, we can induce all necessary components to specify and compute the EPCA objective and decompression.

**Example:**
```julia
F(x) = x * log(x) - x
poisson_epca = EPCA(indim, outdim, F, Val((:F)))
```

### 4. Using $G$ and $g$

Alternatively, we can express the EPCA objective using the log-partition function $G$ and the link function $g$. Starting from the first dervition (and dropping the constant), we have:

```math
\begin{aligned}
-F(g(\Theta)) - \Theta(X - g(\Theta)) &= G(\Theta) - g(\Theta) \Theta - \Theta(X - g(\Theta)) \\
&= G(\Theta) - \Theta X
\end{aligned}
```

This shows that the EPCA objective simplifies to the negative log-likelihood $G(\Theta) - \Theta X$. Therefore, $G$ and $g$ are sufficient to define the EPCA objecitve and link function.

**Example:**
```julia
G(θ) = exp(θ)
g(θ) = exp(θ)
poisson_epca = EPCA(indim, outdim, G, g, Val((:G, :g)))
```

### 5. Using $G$ Alone

Since $g$ is the derivative of $G$ (i.e., $g = G'$), we can now recover $g$ directly from $G$ via symbolic differentiation. This means that providing $G$ alone is enough to specify both the EPCA objective and the link function $g$.

**Example:**
```julia
G(θ) = exp(θ)
poisson_epca = EPCA(indim, outdim, G, Val((:G)))
```

### 6. Using $B_F$ and $g$

If we already have the Bregman divergence $B_F$ and the link function $g$, specifying the EPCA objective is trivial.

**Example:**
```julia
using Distances

B = Distances.gkl_divergence
g(θ) = exp(θ)
poisson_epca = EPCA(indim, outdim, B, g, Val((:B, :g)))
```

### 7. Using $\tilde{B}$ and $g$

Similarly, using the transformed Bregman divergence $\tilde{B}(p, q) = B_F(p \| g(q))$ along with $g$ is straightforward. Given that $\tilde{B}$ is just $B_F$ evaluated at $g(q)$, and we already have the link function $g$, defining the EPCA objective is almost too obvious to mention. While mathematically plain, this specification is useful in practice to avoid domain errors that make optimization difficult with certain exponential family members (e.g., gamma, negative binomial, Pareto).

**Example:**
```julia
Bg(x, θ) = exp(θ) - x * θ + x * log(x) - x
g(θ) = exp(θ)
poisson_epca = EPCA(indim, outdim, Bg, g, Val((:Bg, :g)))
```

## Conclusion

In summary, the EPCA objective function and the decompression function $g$ can be derived from various components. The flexibility afforded by Julia's multiple dispatch and symbolic differentiation capabilities allows for efficient computation of EPCA in different scenarios. Depending on the form of the data and the problem at hand, one can choose the most convenient set of components to define and compute the EPCA objective.

