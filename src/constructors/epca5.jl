struct EPCA5 <: EPCA
    V::AbstractMatrix{<:Real}

    Bg::Function
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

function _make_loss(epca::EPCA5, X)
    # unpack
    Bg = epca.Bg
    μ = epca.μ
    ϵ = epca.ϵ
    
    # construct EPCA objective
    L(θ) = begin
        divergence = @. Bg(X, θ)
        regularizer = @. Bg(μ, θ)
        loss = sum(@. divergence + ϵ * regularizer)
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

    epca = EPCA5(
        V,
        Bg,
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