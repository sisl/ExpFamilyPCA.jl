# Gaussian EPCA and the Squared Frobenius Norm

We want to show that the squared Frobenius norm $\frac{1}{2} \|A - B \|_F^2$ is a Bregman divergence. Let $\psi(A) = \frac{1}{2}\|A\|_F^2$, so that $\nabla \psi(A) = A$. Using norm properties, we can then write the Bregman divergence associated with $\psi$ as

$$\begin{aligned}
B_\psi(A \| B) &= \psi(A) - \psi(B) - \langle \nabla \psi(B), A - B \rangle \\
&= \frac{1}{2}\|A\|_F^2 - \frac{1}{2}\|B\|_F^2 - \langle B, A \rangle + \langle B, B \rangle \\
&= \frac{1}{2}\|A\|_F^2 - \langle B, A \rangle + \frac{1}{2}\|B\|_F^2 \\
&= \frac{1}{2} \big[ \langle A, A \rangle - 2\langle B, A \rangle + \langle B, B \rangle \big] \\
&= \frac{1}{2} \langle A - B, A - B \rangle \\
&= \frac{1}{2} \| A - B \|_F^2.
\end{aligned}$$

Similarly, the Bregman divergence induced from the log-partition of the Gaussian $G(\theta) = \theta^2/2$ is the squared Euclidean distance.