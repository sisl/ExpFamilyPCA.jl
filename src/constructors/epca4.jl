struct EPCA4 <: EPCA
    Bg::Function  # Bregman divergence composed with the link function in the 2nd slot, that is Bg(⋅, ⋅) = B_F(⋅, g(⋅)).
    g::Function  # link function
    V::AbstractMatrix{<:Real}
    options::Options
end

function _make_loss(epca::EPCA4, X)
    Bg = epca.Bg
    @unpack μ, ϵ = epca.options
    @assert ϵ > 0 "ϵ must be positive."
    
    L(θ) = begin
        divergence = @. Bg(X, θ)
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
    g::Function,
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