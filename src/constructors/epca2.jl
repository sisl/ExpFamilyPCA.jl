struct EPCA2 <: EPCA
    V::AbstractMatrix{<:Real}

    G::Union{Function, FunctionWrapper}  # log-parition function
    g::Union{Function, FunctionWrapper}  # link function

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

    fX = map(X) do x
        fx = _binary_search_monotone(
            g,
            x; 
            tol=tol
        )
        return fx
    end
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
    G::Union{Function, FunctionWrapper},
    g::Union{Function, FunctionWrapper},
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
    _check_common_arguments(indim, outdim, ϵ)

    V = _initialize_V(indim, outdim, V_init)
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
    G::Union{Function, FunctionWrapper},
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
    _check_common_arguments(indim, outdim, ϵ)

    # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
    g = _differentiate(G, metaprogramming)
    V = _initialize_V(indim, outdim, V_init)
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
