struct EPCA4{
    FT1<:Function,
    FT2<:Union{Function, FunctionWrapper},
    MT<:AbstractMatrix{<:Real},
    OT<:Options
} <: EPCA
    Bg::FT1  # Bregman divergence composed with the link function in the 2nd slot, that is Bg(⋅, ⋅) = B_F(⋅, g(⋅)).
    g::FT2  # link function
    V::MT
    options::OT
end

function _make_loss(epca::EPCA4, X)
    Bg = epca.Bg
    (; μ, ϵ) = epca.options
    @assert ϵ >= 0 "ϵ must be non-negative."
    
    L(x, θ) = begin
        divergence = @. Bg(x, θ)
        regularizer = @. Bg(μ, θ)
        loss = mean(@. divergence + ϵ * regularizer)
        return loss
    end
    
    return L
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    Bg::Function,
    g::Union{Function, FunctionWrapper},
    ::Val{(:Bg, :g)};
    options::Options = Options()
)
    V = _initialize_V(indim, outdim, options)
    epca = EPCA4(Bg, g, V, options)
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    Bg::Function,
    G::Function,
    ::Val{(:Bg, :G)};
    options::Options = Options()
)
    g = _differentiate(G, options.metaprogramming)
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = options
    )
    return epca
end
