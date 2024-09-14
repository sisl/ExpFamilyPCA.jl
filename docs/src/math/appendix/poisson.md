# Poisson EPCA and the Generalized KL-Divergence

The cumulant of the Poisson distribution is $G(\theta) = e^\theta$, so the the link function (its derivative) is $g(\theta) = e^\theta$. From the [appendix](./inverses.md), we know that $f(x) = g^{-1}(x) = -\log{x}$ and 

$$\begin{aligned}
F(x) 
&= \theta \cdot x - G(\theta) \\
&= f(x) \cdot x - G(f(x)) \\
&= x \log{x} - x.
\end{aligned}$$

The Bregman divergence induced from $F$ is

$$\begin{aligned}
B_F(p \| q) 
&= F(p) - F(q) - \langle f(q), p - q \rangle \\
&= p \log p - p - q \log q + q - \langle \log q, p - q \rangle \\
&= p \log \frac{p}{q} - p + q.
\end{aligned}$$

so $B_F$ is generalized KL-divergence.
