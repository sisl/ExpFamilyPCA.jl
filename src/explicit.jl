mutable struct ExplicitEPCA <: EPCA
    V
    Bregman::Function  # Bregman divergence
    g::Function  # link function
    μ
end


function PoissonEPCA()
    # assumes χ = ℤ
    ϵ = eps()
    @. begin
        Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p
        g(θ) = exp(θ)
    end
    μ = g(0)
    ExplicitEPCA(missing, Bregman, g, μ)
end


function BernoulliEPCA()
    # assumes χ = {0, 1}
    ϵ = eps()
    @. begin
        Bregman(p, q) = p * log((p + ϵ)/ (q + ϵ)) + (1 - p) * log((1 - p + ϵ) / (1 - q + ϵ))
        g(θ) = exp(θ) / (1 + exp(θ))
    end
    μ = g(1)
    ExplicitEPCA(missing, Bregman, g, μ)
end


function NormalEPCA()
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    @. begin
        Bregman(p, q) = (p - q)^2 / 2
        g(θ) = θ
    end
    μ = g(0)
    ExplicitEPCA(missing, Bregman, g, μ)
end


function fit!(
    epca::ExplicitEPCA, 
    X;
    μ=epca.μ,
    maxoutdim=1, 
    maxiter=100,
    verbose=false,
    ϵ=eps()
)
    B, g = epca.Bregman, epca.g
    L(Θ) = begin
        X̂ = g.(Θ)
        sum(B(X, X̂) + ϵ * B(μ, X̂))
    end    
    n, d = size(X)
    A = ones(n, maxoutdim)
    V = ones(maxoutdim, d)
    for _ in 1:maxiter
        V = Optim.minimizer(optimize(V̂->L(A * V̂), V))
        A = Optim.minimizer(optimize(Â->L(Â * V), A))
        if verbose
            @show L(A * V)
        end
    end
    epca.V = V
    return A
end


function compress(
    epca::ExplicitEPCA, 
    X;
    μ=epca.μ,
    maxoutdim=1, 
    maxiter=100,
    verbose=false,
    ϵ=eps()
)
    B, g, V = epca.Bregman, epca.g, epca.V
    L(Θ) = begin
        X̂ = g.(Θ)
        sum(@. B(X, X̂) + ϵ * B(μ, X̂))
    end    
    n, _ = size(X)
    A = ones(n, maxoutdim)
    for _ in 1:maxiter
        A = Optim.minimizer(optimize(Â->L(Â * V), A))
        if verbose
            @show L(A * V)
        end
    end
    return A
end

decompress(epca::ExplicitEPCA, A) = epca.g(A * epca.V)
