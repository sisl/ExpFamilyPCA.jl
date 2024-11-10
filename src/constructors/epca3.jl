struct EPCA3{
    FT1<:Union{Function, FunctionWrapper, PreMetric},
    FT2<:Union{Function, FunctionWrapper},
    MT<:AbstractMatrix{<:Real},
    OT<:Options
} <: EPCA
    B::FT1  # Bregman divergence
    g::FT2  # link function
    V::MT
    options::OT
end

function _make_loss(epca::EPCA3, X)
    (; B, g) = epca
    (; μ, ϵ) = epca.options
    @assert ϵ >= 0 "ϵ must be non-negative."

    L(x, θ) = begin
        gθ = g.(θ)  # think of this as X̂
        divergence = @. B(x, gθ)
        regularizer = @. B(μ, gθ)
        loss = mean(@. divergence + ϵ * regularizer)
        return loss
    end
    return L
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    B::Union{Function, FunctionWrapper, PreMetric},
    g::Union{Function, FunctionWrapper},
    ::Val{(:B, :g)};
    options = Options()
)
    V = _initialize_V(indim, outdim, options)
    epca = EPCA3(B, g, V, options)
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    B::Union{Function, FunctionWrapper, PreMetric},
    G::Union{Function, FunctionWrapper},
    ::Val{(:B, :G)};
    options = Options()
)
    g = _differentiate(G, options.metaprogramming)
    epca = EPCA(
        indim,
        outdim,
        B,
        g,
        Val((:B, :g));
        options = options
    )
    return epca
end
