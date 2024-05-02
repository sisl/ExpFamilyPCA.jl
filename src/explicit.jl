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
        Bregman(p, q) = p * log((p + epsilon) / (q + epsilon)) + q - p
        g(theta) = exp(theta)
    end
    mu = g(0)
    EPCA(Bregman, g, mu)
end


function BernoulliEPCA()
    # assumes X = {0, 1}
    epsilon = eps()
    @. begin
        Bregman(p, q) = p * log((p + epsilon)/ (q + epsilon)) + (1 - p) * log((1 - p + epsilon) / (1 - q + epsilon))
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


function fit!(
    epca::ExplicitEPCA, 
    X;
    mu=epca.mu,
    maxoutdim=1, 
    maxiter=1000,
    verbose=false,
    print_steps=10,
    epsilon=eps(),
)
    B, g = epca.Bregman, epca.g
    L(theta) = begin
        X_hat= g.(theta)
        sum(B(X, X_hat) + epsilon * B(mu, X_hat))
    end    
    n, d = size(X)
    A = ones(n, maxoutdim)
    V = ismissing(epca.V) ? ones(maxoutdim, d) : epca.V
    for i in 1:maxiter
        V = Optim.minimizer(optimize(V_hat->L(A * V_hat), V))
        result = optimize(A_hat->L(A_hat * V), A)
        loss = Optim.minimum(result)
        A = Optim.minimizer(result)
        if verbose && (i % print_steps == 0 || i == 1)
            println("Iteration: $i/$maxiter | Loss: $loss")
        end
    end
    epca.V = V
    return A
end


function compress(
    epca::ExplicitEPCA, 
    X;
    mu=epca.mu,
    maxoutdim=1, 
    maxiter=100,
    verbose=false,
    print_stesp=10,
    epsilon=eps()
)
    B, g, V = epca.Bregman, epca.g, epca.V
    L(theta) = begin
        X_hat= g.(theta)
        sum(@. B(X, X_hat) + epsilon * B(mu, X_hat))
    end    
    n, _ = size(X)
    A = ones(n, maxoutdim)
    for _ in 1:maxiter
        A = Optim.minimizer(optimize(A_hat->L(A_hat * V), A))
        if verbose && (i % print_steps == 0 || i == 1)
            println("Iteration: $i/$maxiter | Loss: $loss")
        end
    end
    return A
end

decompress(epca::ExplicitEPCA, A) = epca.g(A * epca.V)
