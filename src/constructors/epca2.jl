struct EPCA2 <: EPCA
    V::AbstractMatrix{<:Real}

    G::Function
    g::Function  # link function

    # hyperparameters
    tol::Real
    μ::Real
    ϵ::Real

    A_init_value::Union{Real, Nothing}
    A_lower::Union{Real, Nothing}
    A_upper::Union{Real, Nothing}
    V_lower::Union{Real, Nothing}
    V_upper::Union{Real, Nothing}
end

function _make_loss(epca::EPCA2, X)
    # unpack
    G = epca.G
    g = epca.g
    tol = epca.tol
    μ = epca.μ
    ϵ = epca.ϵ

    fX = _binary_search_monotone.(
        g,
        X; 
        tol=tol
    )
    zX = @. fX * X - G(fX)
    
    fμ = _binary_search_monotone(
        g, 
        μ; 
        tol=0
    )  # NOTE: μ is scalar, so we can have very low tol
    zμ = fμ * μ - G(fμ)

    L(θ) = begin
        Gθ = G.(θ)
        divergence = @. zX + Gθ - θ * X
        regularizer = @. zμ + Gθ - θ * μ
        loss = sum(@. divergence + ϵ * regularizer)
        return loss
    end
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    g::Function,
    ::Val{(:G, :g)};
    tol = eps(),
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

    epca = EPCA2(
        V,
        G,
        g,
        tol,
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


"""
 # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    ::Val{(:G)};
    tol = eps(),
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

    # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
    # G induces g, Fg = F(g(θ)), and fg = f(g(θ))
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
        G,
        g,
        Val((:G, :g));
        tol = tol,
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
