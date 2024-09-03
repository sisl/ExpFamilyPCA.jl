struct EPCA3 <: EPCA
    B::Union{Function, FunctionWrapper, PreMetric}  # Bregman divergence
    g::Union{Function, FunctionWrapper}  # link function
    V::AbstractMatrix{<:Real}
    options::Options
end

function _make_loss(epca::EPCA3, X)
    @unpack B, g = epca
    @unpack μ, ϵ = epca.options
    @assert ϵ > 0 "ϵ must be positive."

    L(θ) = begin
        gθ = g.(θ)  # think of this as X̂
        divergence = @. B(X, gθ)
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
