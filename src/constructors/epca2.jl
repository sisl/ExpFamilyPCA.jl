struct EPCA2{
    FT1<:Union{Function, FunctionWrapper},
    FT2<:Union{Function, FunctionWrapper},
    MT<:AbstractMatrix{<:Real},
    OT<:Options
} <: EPCA
    G::FT1  # log-parition function
    g::FT2  # link function
    V::MT
    options::OT
end

function _make_loss(epca::EPCA2, X)
    (; G, g) = epca
    (; tol, μ, ϵ) = epca.options
    @assert ϵ >= 0 "ϵ must be non-negative."

    L(x, θ) = begin
        Gθ = G.(θ)
        divergence = @. Gθ - θ * x
        regularizer = @. Gθ - θ * μ
        loss = mean(@. divergence + ϵ * regularizer)
        return loss
    end
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Union{Function, FunctionWrapper},
    g::Union{Function, FunctionWrapper},
    ::Val{(:G, :g)};
    options = Options()
)
    V = _initialize_V(indim, outdim, options)
    epca = EPCA2(G, g, V, options)
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Union{Function, FunctionWrapper},
    ::Val{(:G)};
    options = Options()
)
    g = _differentiate(G, options.metaprogramming)
    epca = EPCA(
        indim,
        outdim,
        G,
        g,
        Val((:G, :g));
        options = options
    )
    return epca
end
