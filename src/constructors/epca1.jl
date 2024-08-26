struct EPCA1 <: EPCA
    V::AbstractMatrix{<:Real}

    F::Function
    g::Function

    # hyperparameters
    μ::Real
    ϵ::Real

    A_init_value::Union{Real, Nothing}
    A_lower::Union{Real, Nothing}
    A_upper::Union{Real, Nothing}
    V_lower::Union{Real, Nothing}
    V_upper::Union{Real, Nothing}
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
        gθ = g.(θ)
        z = @. F(gθ) - gθ * θ
        divergence = @. FX - z - X * θ
        regularizer = @. Fμ - z - μ * θ
        loss = sum(@. divergence + ϵ * regularizer)
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
    μ = 1,
    ϵ = eps(),
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = nothing,
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert ϵ > 0 "ϵ must be positive."

    if isnothing(V_init)
        V = ones(outdim, indim)
    else
        @assert size(V_init) == (outdim, indim) "V_init must have dimensions (outdim, indim)."
        V = V_init
    end
    
    epca = EPCA1(
        V,
        F,
        g,
        μ,
        ϵ,
        A_init_value,
        A_lower,
        A_upper,
        V_lower,
        V_upper
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    f::Function,
    ::Val{(:F, :f)};
    μ = 1,
    ϵ = eps(),
    low = -1e10,
    high = 1e10,
    tol = 1e-10,
    maxiter = 1e6,
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = nothing,
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert ϵ > 0 "ϵ must be positive."
    @assert low < high "Low bound (low) must be less than high bound (high)."
    @assert tol > 0 "Tolerance (tol) must be a positive number."
    @assert maxiter > 0 "Maximum iterations (maxiter) must be a positive number."

    g = _invert_legendre(
        f;
        low = low,
        high = high,
        tol = tol,
        maxiter = maxiter
    )

    # Initialize V
    if isnothing(V_init)
        V = ones(outdim, indim)
    else
        @assert size(V_init) == (outdim, indim) "V_init must have dimensions (outdim, indim)."
        V = V_init
    end

    epca = EPCA1(
        V,
        F,
        g,
        μ,
        ϵ,
        A_init_value,
        A_lower,
        A_upper,
        V_lower,
        V_upper
    )
    return epca
end


function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    ::Val{(:F)};
    μ = 1,
    ϵ = eps(),
    metaprogramming = true,
    low = -1e10,
    high = 1e10,
    tol = 1e-10,
    maxiter = 1e6,
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = nothing,
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert ϵ > 0 "ϵ must be positive."
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

    # Initialize V
    if isnothing(V_init)
        V = ones(outdim, indim)
    else
        @assert size(V_init) == (outdim, indim) "V_init must have dimensions (outdim, indim)."
        V = V_init
    end

    epca = EPCA(
        indim,
        outdim,
        F,
        f,
        Val((:F, :f));
        μ = μ,
        ϵ = ϵ,
        low = low, 
        high = high, 
        tol = tol, 
        maxiter = maxiter,
        V_init = V,
        A_init_value = A_init_value,
        A_lower = A_lower,
        A_upper = A_upper,
        V_lower = V_lower,
        V_upper = V_upper
    )
    return epca
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    G::Function,
    ::Val{(:F, :G)};
    μ = 1,
    ϵ = eps(),
    metaprogramming = true,
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = nothing,
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert ϵ > 0 "ϵ must be positive."

    @variables θ
    D = Differential(θ)
    _g = expand_derivatives(D(G(θ)))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    # Initialize V
    if isnothing(V_init)
        V = ones(outdim, indim)
    else
        @assert size(V_init) == (outdim, indim) "V_init must have dimensions (outdim, indim)."
        V = V_init
    end

    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g));
        μ = μ,
        ϵ = ϵ,
        V_init = V,
        A_init_value = A_init_value,
        A_lower = A_lower,
        A_upper = A_upper,
        V_lower = V_lower,
        V_upper = V_upper
    )
    return epca
end
