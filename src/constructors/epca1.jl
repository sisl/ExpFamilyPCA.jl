struct EPCA1 <: EPCA
    V::AbstractMatrix{<:Real}

    F::Union{Function, FunctionWrapper}  # Legendre dual of the log-partition
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
    F::Union{Function, FunctionWrapper},
    g::Union{Function, FunctionWrapper},
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
    _check_common_arguments(indim, outdim, ϵ)

    V = _initialize_V(indim, outdim, V_init)
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
    F::Union{Function, FunctionWrapper},
    f::Union{Function, FunctionWrapper},
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
    _check_common_arguments(indim, outdim, ϵ)
    _check_binary_search_arguments(low, high, tol, maxiter)
    @assert isfinite(f(μ)) "μ must be in the range of g meaning f(μ) should be finite."

    g = _invert_legendre(
        f;
        low = low,
        high = high,
        tol = tol,
        maxiter = maxiter
    )
    V = _initialize_V(indim, outdim, V_init)
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
    F::Union{Function, FunctionWrapper},
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
    _check_common_arguments(indim, outdim, ϵ)
    _check_binary_search_arguments(low, high, tol, maxiter)  

    f = _differentiate(F, metaprogramming)
    @assert isfinite(f(μ)) "μ must be in the range of g meaning f(μ) should be finite."

    V = _initialize_V(indim, outdim, V_init)
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
    F::Union{Function, FunctionWrapper},
    G::Union{Function, FunctionWrapper},
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
    _check_common_arguments(indim, outdim, ϵ)

    g = _differentiate(G, metaprogramming)
    V = _initialize_V(indim, outdim, V_init)
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
