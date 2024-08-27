struct EPCA3 <: EPCA
    V::AbstractMatrix{<:Real}

    Bregman::Union{Function, FunctionWrapper, PreMetric}  # Bregman divergence can be specified using a Julia Union{Function, FunctionWrapper} or a Distances.jl PreMetric
    g::Union{Function, FunctionWrapper}  # g function

    # hyperparameters
    μ::Real
    ϵ::Real

    A_init_value::Union{Real, Nothing}
    A_lower::Union{Real, Nothing}
    A_upper::Union{Real, Nothing}
    V_lower::Union{Real, Nothing}
    V_upper::Union{Real, Nothing}
end

function _make_loss(epca::EPCA3, X)
    B = epca.Bregman
    g = epca.g
    μ = epca.μ
    ϵ = epca.ϵ
    L(θ) = begin
        gθ = g.(θ)  # think of this as X̂
        divergence = @. B(X, gθ) + ϵ * B(μ, gθ)
        return sum(divergence)
    end
    return L
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    Bregman::Union{Function, FunctionWrapper, PreMetric},
    g::Union{Function, FunctionWrapper},
    ::Val{(:Bregman, :g)};
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

    # Initialize V
    if isnothing(V_init)
        V = ones(outdim, indim)
    else
        @assert size(V_init) == (outdim, indim) "V_init must have dimensions (outdim, indim)."
        V = V_init
    end

    epca = EPCA3(
        V,
        Bregman,
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
    Bregman::Union{Function, FunctionWrapper, PreMetric},
    G::Union{Function, FunctionWrapper},
    ::Val{(:Bregman, :G)};
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
    G_expr = G(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(G_expr))

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
        Bregman,
        g,
        Val((:Bregman, :g));
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
