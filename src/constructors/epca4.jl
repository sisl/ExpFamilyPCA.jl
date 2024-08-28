struct EPCA4 <: EPCA
    V::AbstractMatrix{<:Real}

    f::Union{Function, FunctionWrapper}  # gradient of the Legendre dual of the log-partition
    G::Union{Function, FunctionWrapper}  # log-partition function
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

function _make_loss(epca::EPCA4, X)
    # unpack
    f = epca.f
    G = epca.G
    μ = epca.μ
    ϵ = epca.ϵ

    fX = f.(X)
    zX = @. fX * X - G(fX)
    
    fμ = f.(μ)
    zμ = @. fμ * μ - G(fμ)

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
    f::Union{Function, FunctionWrapper},
    G::Union{Function, FunctionWrapper},
    g::Union{Function, FunctionWrapper},
    ::Val{(:f, :G, :g)};
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
    epca = EPCA4(
        V,
        f,
        G,
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
    f::Union{Function, FunctionWrapper},
    G::Union{Function, FunctionWrapper},
    ::Val{(:f, :G)};
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

    # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
    g = _differentiate(G, metaprogramming)
    V = _initialize_V(indim, outdim, V_init)
    epca = EPCA(
        indim,
        outdim,
        f,
        G,
        g,
        Val((:f, :G, :g));
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
