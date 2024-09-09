struct EPCA1 <: EPCA
    F::Union{Function, FunctionWrapper}  # Legendre dual of the log-partition
    g::Union{Function, FunctionWrapper}  # link function
    V::AbstractMatrix{<:Real}
    options::Options
end

function _make_loss(epca::EPCA1, X)
    @unpack F, g = epca
    @unpack μ, ϵ = epca.options
    @assert ϵ > 0 "ϵ must be positive."

    # construct EPCA objective
    # L(θ) = begin
    #     gθ = g.(θ)
    #     z = @. F(gθ) - gθ * θ
    #     divergence = @. -z - X * θ
    #     regularizer = @. -z - μ * θ
    #     loss = mean(@. divergence + ϵ * regularizer)
    #     return loss
    # end

    L(x, θ) = begin
        gθ = g.(θ)
        z = @. F(gθ) - gθ * θ
        divergence = @. -z - x * θ
        regularizer = @. -z - μ * θ
        loss = mean(@. divergence + ϵ * regularizer)
        return loss
    end
    return L
end


function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Union{Function, FunctionWrapper},
    g::Union{Function, FunctionWrapper},
    ::Val{(:F, :g)};
    options = Options()
)
    V = _initialize_V(indim, outdim, options)
    epca = EPCA1(F, g, V, options)
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Union{Function, FunctionWrapper},
    f::Union{Function, FunctionWrapper},
    ::Val{(:F, :f)};
    options = Options()
)
    @assert isfinite(f(options.μ)) "μ must be in the range of g meaning f(μ) should be finite."
    @unpack low, high, tol, maxiter = options
    g = _invert_legendre(f, options)
    V = _initialize_V(indim, outdim, options)
    epca = EPCA1(F, g, V, options)
    return epca
end


function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Union{Function, FunctionWrapper},
    ::Val{(:F)};
    options = Options()
)
    f = _differentiate(F, options.metaprogramming)
    @assert isfinite(f(options.μ)) "μ must be in the range of g meaning f(μ) should be finite."
    epca = EPCA(
        indim,
        outdim,
        F,
        f,
        Val((:F, :f));
        options = options
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Union{Function, FunctionWrapper},
    G::Union{Function, FunctionWrapper},
    ::Val{(:F, :G)};
    options::Options = Options()
)
    g = _differentiate(G, options.metaprogramming)
    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g));
        options = options
    )
    return epca
end
