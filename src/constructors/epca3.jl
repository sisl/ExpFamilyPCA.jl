struct EPCA3 <: EPCA
    V::AbstractMatrix{<:Real}

    B::Union{Function, FunctionWrapper, PreMetric}  # Bregman divergence
    g::Union{Function, FunctionWrapper}  # link function

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
    B = epca.B
    g = epca.g
    μ = epca.μ
    ϵ = epca.ϵ
    L(θ) = begin
        gθ = g.(θ)  # think of this as X̂
        divergence = @. B(X, gθ) + ϵ * B(μ, gθ)
        if isnan(sum(divergence))
            @infiltrate
        end
        return sum(divergence)
    end
    return L
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    B::Union{Function, FunctionWrapper, PreMetric},
    g::Union{Function, FunctionWrapper},
    ::Val{(:B, :g)};
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

    epca = EPCA3(
        V,
        B,
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
    B::Union{Function, FunctionWrapper, PreMetric},
    G::Union{Function, FunctionWrapper},
    ::Val{(:B, :G)};
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
    _check_common_arguments(indim, outdim, ϵ)

    g = _differentiate(G, metaprogramming)
    V = _initialize_V(indim, outdim, V_init)
    epca = EPCA(
        indim,
        outdim,
        B,
        g,
        Val((:B, :g));
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
