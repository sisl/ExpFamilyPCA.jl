# ExpFamilyPCA

[![Build Status](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Objective

The generalized Bregman divergence is

$$
B_F(p \| q) = F(p) - F(q) - f(q)(p - q).
$$

Where $f(x) = F'(x)$. We want to minimize the Bregman divergence between our $n \times d$ data matrix $X$ and our compression parameters $\Theta = AV$ where $A \in \mathbb{R}^{n \times l}$ is our low-dimensional representation and $V \in \mathbb{R}^{l \times d}$ is our transformation matrix. If $g(x) = G'(x)$ is our link function (and $G$ is any differentiable convex function), then we want to minimize

$$
L(V, A) = B_F(X \| g(\Theta)) + \epsilon B_F(\mu \| g(\Theta))
$$

where the second term is a stabilizer, $\epsilon$ is some small positive constant, and $\mu$ is any value in the range of $g$. 

Let $F_g = F \circ g$ and let $f_g = f \circ g$. It can be shown that $f(x) = g^{-1}(x)$ under mild conditions. Then, we can write the first term as 

$$
B_F(X \| g(\Theta)) = F(X) - F_g(\Theta) - f_g(\Theta) (X - g(\Theta)).
$$

Similarly, we can rewrite the second term as

$$
\epsilon B_F(\mu \| g(\Theta)) = \epsilon F(\mu) - \epsilon F_g(\Theta) - \epsilon f_g(\Theta) (\mu - g(\Theta)).
$$

We are given $G$. From $G$, we can use symbolic differentiation to easily and programmatically find $g$ (thanks to Julia's multiple dispatch and metaprogramming). Since $F$ is defined parametrically, we know $F_g$ and $f_g$; however, defining $F$ directly is difficult because we would have to solve an equation with symbolic manipulation and Julia currently isn't good at this. Luckily, our objective doesn't require us to know $F$ symbolically, since we're only asked for $F(X)$ and $F(\mu)$ which are both constants. It thus suffices to simply *evaluate* $F$ at $X$ and $\mu$. To do so, first recall that $F$ is defined as

$$
F(\omega) \equiv \theta \cdot g(\theta) - G(\theta)
$$

where $g(\theta) = \omega$ and $\theta = g^{-1}(\omega)$. We can then write $F$ directly in terms of $\omega$ as

$$
F(\omega) = g^{-1}(\omega) \cdot \omega - G(g^{-1}(\omega)).
$$

Since $g$ is monotone (since $G$ is convex and differentiable), not only do we know that the inverse *exists*, but we can also find it efficiently with a binary search. 

Explicitly, we can frame our problem like this. Given $g$, evaluate an unknown $g^{-1}$ and some point $a$. Let $b = g^{-1}(a)$, then $g(b) = g(g^{-1}(a)) = a$, so we simply need to find $b$ such that $g(b) = a$ and then $b = g^{-1}(g(b)) = g^{-1}(a)$. Since $g$ is monotone, we can quickly find such a $b$ by searching for the condition that $g(b) = a$.

Since we only need to evaluate $F(X)$ and $F(\mu)$, this means we need to find $b, b'$ such that $g(b) = X$ and $g(b') = \mu$.
