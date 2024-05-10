struct ExplicitEPCA <: EPCA
    V
    Bregman  # Bregman divergence
    g::Function  # link function

    # hyperparameters
    μ
    ϵ
end

"""Explicitly specify the Bregman divergence."""
function EPCA(indim, outdim, Bregman::Function, g::Function; μ=1, ϵ=eps())
    ExplicitEPCA(ones(outdim, indim), Bregman, g, μ, ϵ)
end


"""Induces the Bregman divergence from F and f."""
function EPCA(indim, outdim, F::Function, f::Function, g::Function; μ=1, ϵ=eps())
    ExplicitEPCA(ones(outdim, indim), Distances.Bregman(F, f), g, μ, ϵ)
end

# TODO: could create EPCA entirely from F

function PoissonEPCA(indim, outdim; ϵ=eps())
    # assumes χ = ℤ
    @. begin
        F(x) = x * log(x + ϵ) - x
        f(x) = log(x + ϵ)
        g(θ) = exp(θ)
    end
    μ = g(0)
    EPCA(indim, outdim, F, f, g; μ=μ, ϵ=ϵ)
end


function BernoulliEPCA(indim, outdim; ϵ=eps())
    # assumes χ = {0, 1}
    @. begin
        F(x) = x * log(x + ϵ) + (1 - x) * log(1 - x + ϵ)
        f(x) = log(x + ϵ) - log(1 - x + ϵ)
        g(x) = exp(x) / (1 + exp(x))
    end
    μ = g(0)
    EPCA(indim, outdim, F, f, g; μ=μ, ϵ=ϵ)
end


function NormalEPCA(indim, outdim; ϵ=eps())
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    @. begin
        Bregman(p, q) = (p - q)^2 / 2
        g(θ) = θ
    end
    μ = g(1)
    EPCA(indim, outdim, Bregman, g; μ=μ, ϵ=ϵ)
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