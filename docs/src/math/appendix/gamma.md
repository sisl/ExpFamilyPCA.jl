# Gamma EPCA and the Itakura-Saito Distance

The cumulant of the gamma distribution is $G(\theta) = -\log(-\theta)$, so the the link function (its derivative) is $g(\theta) = \nabla_\theta G(\theta) = -\frac{1}{\theta}$. From the [appendix](./inverses.md), we know that $f(x) = g^{-1}(x) = -\frac{1}{x}$ and 

$$\begin{aligned}
F(x) 
&= \theta \cdot x - G(\theta) \\
&= f(x) \cdot x - G(f(x)) \\
&= -1 - \log(x).
\end{aligned}$$

The Bregman divergence induced from $F$ is

$$\begin{aligned}
B_F(p \| q) 
&= F(p) - F(q) - \langle f(q), p - q \rangle \\
&= -1 - \log p + 1 + \log q + \Big\langle \frac{1}{q}, p - q \Big\rangle \\
&= \frac{p}{q} - \log \frac{p}{q} - 1,
\end{aligned}$$

so $B_F$ is the Itakura-Saito [ItakuraSaito](@cite) distance as desired. Further, the EPCA objective is

$$\begin{aligned}
B_F(x \| g(\theta)) = \frac{p}{g(\theta)} - \log \frac{p}{g(\theta)} - 1 = -p\theta - \log(-p\theta) - 1.
\end{aligned}$$
