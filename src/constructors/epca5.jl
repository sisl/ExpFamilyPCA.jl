struct EPCA5 <: EPCA
    V::AbstractMatrix{<:Real}

    Bg::Function  # Bregman divergence composed with the link function in the 2nd slot, that is Bg(⋅, ⋅) = B_F(⋅, g(⋅)).
    g::Function  # link function

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
    _check_common_arguments(indim, outdim, ϵ)
    V = _initialize_V(indim, outdim, V_init)
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