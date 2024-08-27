Recall, a Bregman divergence $D_b$ is defined with respect to a convex function $b$, by 

$$
D_b(u, v) = b(u) - b(v) - \langle \nabla b(v), u - v \rangle
$$

where $\langle , \rangle$ denotes an inner product. You can interpret it as the difference between $b(u)$ and its first-order Taylor approximation at $u$, when that Taylor approximation is made around the point $v$. 

Now suppose you have data $y$ from an exponential family model with natural parameter $\theta$. (Note: in a GLM, we would take $\theta = X \beta$.) Thus we can write the density as 

$$
p(y; \theta, \phi) = \exp\Bigg( \frac{\langle y, \theta \rangle - A(\theta)}{\phi} \Bigg)  f(y, \phi)
$$

Here $A$ is the log partition function, and $\phi$ is a dispersion parameter, which you can just ignore from now on. Let $\ell(\theta)$ be the log likelihood. Then one can check that the deviance is:

$$
d(\theta, \theta^*) = 2(\ell(\theta) - \ell(\theta^*)) = 2(A(\theta) - A(\theta^*) - \langle y, \theta - \theta^* \rangle)
$$

where $\theta^*$ is the parameter in the saturated model. We can define it by solving $\nabla A(\theta^*) = y$, because that means that in this parametrization, we'll be able to match the mean exactly: $\mathbb{E}[y] = \mathbb{E}[\nabla A(\theta^*)]$.

So then we can write 

$$
d(\theta, \theta^*) = 2(A(\theta) - A(\theta^*) - \langle\nabla A(\theta^*), \theta - \theta^*\rangle) 
$$

But this is simply the Bregman divergence $D_b(\theta, \theta^*)$ with $b = 2A$. 

In conclusion, maximum likelihood in this model is equivalent to minimizing the deviance which is equivalent to minimizing the Bregman divergence defined by twice the log partition function.

# TODO
- [ ] show properties of Bregman like g is the inverse of f when F is defined as it is