mutable struct ExplicitEPCA <: EPCA
    V
    Bregman::Function  # Bregman divergence
    g::Function  # link function
    mu
end

function EPCA(Bregman::Function, g::Function, mu)
    ExplicitEPCA(missing, Bregman, g, mu)
end

function PoissonEPCA()
    # assumes X = {integers}
    epsilon = eps()
    @. begin
        Bregman(p, q) = p * (log(p + epsilon) - log(q + epsilon)) + q - p
        g(theta) = exp(theta)
    end
    mu = g(0)
    EPCA(Bregman, g, mu)
end


function BernoulliEPCA()
    # assumes X = {0, 1}
    epsilon = eps()
    @. begin
        Bregman(p, q) = p * (log(p + epsilon) - log(q + epsilon)) + (1 - p) * (log(1 - p + epsilon) - log(1 - q + epsilon))
        g(theta) = exp(theta) / (1 + exp(theta))
    end
    mu = g(1)
    EPCA(Bregman, g, mu)
end


function NormalEPCA()
    # NOTE: equivalent to generic PCA
    # assume X = {reals}
    @. begin
        Bregman(p, q) = (p - q)^2 / 2
        g(theta) = theta
    end
    mu = g(0)
    EPCA(Bregman, g, mu)
end


function _make_loss(epca::ExplicitEPCA, X, epsilon, mu)
    B, g = epca.Bregman, epca.g
    L(theta) = begin
        X_hat = g.(theta)
        sum(@. B(X, X_hat) + epsilon * B(mu, X_hat))
    end
    return L
end


function fit!(
    epca::ExplicitEPCA, 
    X;
    mu=epca.mu,
    maxoutdim=1, 
    maxiter=10,
    verbose=false,
    steps_per_print=10,
    epsilon=eps(),
)
    L = _make_loss(epca, X, epsilon, mu)
    A =  _fit!(epca, X, maxoutdim, L, maxiter, verbose, steps_per_print)
    return A
end


function compress(
    epca::ExplicitEPCA, 
    X;
    mu=epca.mu,
    maxiter=10,
    verbose=false,
    steps_per_print=10,
    epsilon=eps()
)
    L = _make_loss(epca, X, epsilon, mu)
    A = _compress(epca, X, L, maxiter, verbose, steps_per_print)
    return A
end

decompress(epca::ExplicitEPCA, A) = epca.g(A * epca.V)
