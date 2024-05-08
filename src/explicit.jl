mutable struct ExplicitEPCA <: EPCA
    V
    Bregman::Function  # Bregman divergence
    g::Function  # link function

    # hyperparameters
    μ
    ϵ
end


function EPCA(Bregman::Function, g::Function, μ, ϵ=eps())
    ExplicitEPCA(missing, Bregman, g, μ, ϵ)
end


function PoissonEPCA(; ϵ=eps())
    # assumes χ = ℤ
    @. begin
        Bregman(p, q) = p * (log(p + ϵ) - log(q + ϵ)) + q - p
        g(θ) = exp(θ)
    end
    μ = g(0)
    EPCA(Bregman, g, μ, ϵ)
end


function BernoulliEPCA(; ϵ=eps())
    # assumes χ = {0, 1}
    @. begin
        Bregman(p, q) = p * (log(p + ϵ) - log(q + ϵ)) + (1 - p) * (log(1 - p + ϵ) - log(1 - q + ϵ))
        g(θ) = exp(θ) / (1 + exp(θ))
    end
    μ = g(1)
    EPCA(Bregman, g, μ, ϵ)
end


function NormalEPCA(; ϵ=eps())
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    @. begin
        Bregman(p, q) = (p - q)^2 / 2
        g(θ) = θ
    end
    μ = g(0)
    EPCA(Bregman, g, μ, ϵ)
end


function _make_loss(epca::ExplicitEPCA, X)
    B, g, μ, ϵ = epca.Bregman, epca.g, epca.μ, epca.ϵ
    L(θ) = begin
        X̂ = g.(θ)
        divergence = @. B(X, X̂) + ϵ * B(μ, X̂)
        return sum(divergence)
    end
    return L
end