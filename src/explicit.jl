mutable struct ExplicitEPCA <: EPCA
    V
    Bregman::Function  # Bregman divergence
    g::Function  # link function

    # hyperparameters
    mu
    epsilon
end


function EPCA(Bregman::Function, g::Function, mu, epsilon=eps())
    ExplicitEPCA(missing, Bregman, g, mu, epsilon)
end


function PoissonEPCA(; epsilon=eps())
    # assumes X = {integers}
    @. begin
        Bregman(p, q) = p * (log(p + epsilon) - log(q + epsilon)) + q - p
        g(theta) = exp(theta)
    end
    mu = g(0)
    EPCA(Bregman, g, mu, epsilon)
end


function BernoulliEPCA(; epsilon=eps())
    # assumes X = {0, 1}
    @. begin
        Bregman(p, q) = p * (log(p + epsilon) - log(q + epsilon)) + (1 - p) * (log(1 - p + epsilon) - log(1 - q + epsilon))
        g(theta) = exp(theta) / (1 + exp(theta))
    end
    mu = g(1)
    EPCA(Bregman, g, mu, epsilon)
end


function NormalEPCA(; epsilon=eps())
    # NOTE: equivalent to generic PCA
    # assume X = {reals}
    @. begin
        Bregman(p, q) = (p - q)^2 / 2
        g(theta) = theta
    end
    mu = g(0)
    EPCA(Bregman, g, mu, epsilon)
end


function _make_loss(epca::ExplicitEPCA, X)
    B, g, mu, epsilon = epca.Bregman, epca.g, epca.mu, epca.epsilon
    L(theta) = begin
        X_hat = g.(theta)
        sum(@. B(X, X_hat) + epsilon * B(mu, X_hat))
    end
    return L
end