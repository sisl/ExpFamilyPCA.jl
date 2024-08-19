struct EPCA1 <: EPCA
    V::AbstractMatrix{<:Real}
    F::Function
    g::Function

    # hyperparameters
    μ::Real
    ϵ::Real
end


function _make_loss(epca::EPCA1, X)
    # unpack
    F = epca.F
    g = epca.g
    μ = epca.μ
    ϵ = epca.ϵ
    
    # construct EPCA objective
    FX = F.(X)
    Fμ = F(μ)
    L(θ) = begin
        X̂ = g.(θ)
        Fgθ = F.(X̂)
        BF_X = @. FX - Fgθ - θ * (X - X̂)
        BF_μ = @. Fμ - Fgθ - θ * (μ - X̂)
        divergence = @. BF_X + ϵ * BF_μ
        loss = sum(divergence)
        return loss
    end
    return L
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    g::Function,
    ::Val{(:F, :g)};
    μ=1,
    ϵ=eps(),
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert μ > 0 "μ must be a positive number."
    @assert ϵ >= 0 "ϵ must be nonnegative."

    V = ones(outdim, indim)
    epca = EPCA1(
        V,
        F,
        g,
        μ,
        ϵ
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    f::Function,
    ::Val{(:F, :f)};
    μ=1,
    ϵ=eps(),
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert μ > 0 "μ must be a positive number."
    @assert ϵ >= 0 "ϵ must be nonnegative."
    @assert low < high "Low bound (low) must be less than high bound (high)."
    @assert tol > 0 "Tolerance (tol) must be a positive number."
    @assert maxiter > 0 "Maximum iterations (maxiter) must be a positive number."

    g = _invert_legendre(
        f;
        low=low,
        high=high,
        tol=tol,
        maxiter=maxiter
    )
    V = ones(outdim, indim)
    epca = EPCA1(
        V,
        F,
        g,
        μ,
        ϵ
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    ::Val{(:F)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true,
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert μ > 0 "μ must be a positive number."
    @assert ϵ >= 0 "ϵ must be nonnegative."
    @assert low < high "Low bound (low) must be less than high bound (high)."
    @assert tol > 0 "Tolerance (tol) must be a positive number."
    @assert maxiter > 0 "Maximum iterations (maxiter) must be a positive number."    

    # math
    @variables θ
    _F = F(θ)
    D = Differential(θ)
    _f = expand_derivatives(D(_F))

    if metaprogramming
        f = _symbolics_to_julia(_f)
    else
        f = _symbolics_to_julia(_f, θ)
    end

    epca = EPCA(
        indim,
        outdim,
        F,
        f,
        Val((:F, :f));
        μ=μ,
        ϵ=ϵ,
        low=low, 
        high=high, 
        tol=tol, 
        maxiter=maxiter
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    G::Function,
    ::Val{(:F, :G)};
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
    D = Differential(θ)
    _g = expand_derivatives(D(G(θ)))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g));
        μ=μ,
        ϵ=ϵ
    )
    return epca
end