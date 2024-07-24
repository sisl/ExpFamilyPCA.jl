struct ExplicitEPCA <: EPCA
    V
    Bregman  # Bregman divergence
    g::Function  # link function

    # hyperparameters
    μ
    ϵ
end

"""Explicitly specify the Bregman divergence."""
function EPCA(
    indim, 
    outdim, 
    Bregman::Function, 
    g::Function; 
    μ=1, 
    ϵ=eps()
)
    V = ones(outdim, indim)
    epca = ExplicitEPCA(
        V, 
        Bregman, 
        g, 
        μ,
        ϵ
    )
    return epca
end


"""Induces the Bregman divergence from F and f."""
function EPCA(
    indim, 
    outdim, 
    F::Function, 
    f::Function, 
    g::Function; 
    μ=1, 
    ϵ=eps()
)
    V = ones(outdim, indim)
    Bregman = Distances.Bregman(F, f)
    epca = ExplicitEPCA(
        V,
        Bregman,
        g,
        μ,
        ϵ
    )
    return epca
end


function PoissonEPCA(indim, outdim; ϵ=eps())
    # assumes χ = ℤ
    @. begin
        Bregman(p, q) = Distances.gkl_divergence(p, q)
        g(θ) = exp(θ)
    end
    μ = g(0)
    epca = EPCA(indim, outdim, Bregman, g; μ=μ, ϵ=ϵ)
    return epca
end


function BernoulliEPCA(indim, outdim; ϵ=eps())
    # assumes χ = {0, 1}
    @. begin
        F(x) = x * log(x + ϵ) + (1 - x) * log(1 - x + ϵ)
        f(x) = log(x + ϵ) - log(1 - x + ϵ)
        g(θ) = exp(θ) / (1 + exp(θ))
    end
    μ = g(0)
    epca = EPCA(indim, outdim, F, f, g; μ=μ, ϵ=ϵ)
    return epca
end


function NormalEPCA(indim, outdim; ϵ=eps())
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    @. begin
        Bregman(p, q) = Distances.sqeuclidean(p, q) / 2
        g(θ) = θ
    end
    μ = g(1)
    epca = EPCA(indim, outdim, Bregman, g; μ=μ, ϵ=ϵ)
    return epca
end


function ItakuraSaitoEPCA(indim, outdim; ϵ=eps())
    @. begin
        F(x) = x * log(x + ϵ)
        f(x) = 1 + log(x + ϵ)
        g(θ) = exp(θ - 1)
    end
    μ = g(1)
    epca = EPCA(indim, outdim, F, f, g; μ=μ, ϵ=ϵ)
    return epca
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