struct EPCA3 <: EPCA
    V::AbstractMatrix{<:Real}
    Bregman::Union{Function, PreMetric}  # Bregman divergence can be specified using a Julia Function or a Distances.jl PreMetric
    g::Function  # g function

    # hyperparameters
    μ::Real
    ϵ::Real
end

function _make_loss(epca::EPCA3, X)
    B = epca.Bregman
    g = epca.g
    μ = epca.μ
    ϵ = epca.ϵ
    L(θ) = begin
        X̂ = g.(θ)
        divergence = @. B(X, X̂) + ϵ * B(μ, X̂)
        return sum(divergence)
    end
    return L
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    Bregman::Union{Function, PreMetric},
    g::Function,
    ::Val{(:Bregman, :g)};
    μ=1,
    ϵ=eps()
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert μ > 0 "μ must be a positive number."
    @assert ϵ >= 0 "ϵ must be nonnegative."

    V = ones(outdim, indim)
    epca = EPCA3(
        V,
        Bregman,
        g,
        μ,
        ϵ
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    Bregman::Function,
    G::Function,
    ::Val{(:Bregman, :G)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert μ > 0 "μ must be a positive number."
    @assert ϵ >= 0 "ϵ must be nonnegative."

    @variables θ
    G = G(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(G))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    epca = EPCA(
        indim,
        outdim,
        Bregman,
        g::Function,
        Val((:Bregman, :g));
        μ=μ,
        ϵ=ϵ
    )
    return epca
end