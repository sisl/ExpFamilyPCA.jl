@with_kw struct Options
    # symbolic calculus 
    metaprogramming::Bool = true

    # loss hyperparameters
    μ::Real = 1
    ϵ::Real = eps()

    A_init_value::Real = 1.0
    A_lower::Union{Real, Nothing} = nothing
    A_upper::Union{Real, Nothing} = nothing
    A_use_sobol::Bool = false

    V_init_value::Real = 1.0
    V_lower::Union{Real, Nothing} = nothing
    V_upper::Union{Real, Nothing} = nothing
    V_use_sobol::Bool = false

    # binary search options
    low = -1e10
    high = 1e10
    tol = 1e-10
    maxiter = 1e6
end

function NegativeDomain(;
    metaprogramming::Bool = true,
    μ::Real = 1,
    ϵ::Real = eps(),
    low = -1e10,
    high = 1e10,
    tol = 1e-10,
    maxiter = 1e6,
)
    options = Options(
        metaprogramming = metaprogramming,
        μ = μ,
        ϵ = ϵ,
        low = low,
        high = high,
        tol = tol,
        maxiter = maxiter,
        A_init_value = -1,
        A_upper = -1e-4,
        V_init_value = 1,
        V_lower = 1e-4,
    )
    return options
end

function PositiveDomain(
    metaprogramming::Bool = true,
    μ::Real = 1,
    ϵ::Real = eps(),
    low = -1e10,
    high = 1e10,
    tol = 1e-10,
    maxiter = 1e6,
)
    options = Options(
        metaprogramming = metaprogramming,
        μ = μ,
        ϵ = ϵ,
        low = low,
        high = high,
        tol = tol,
        maxiter = maxiter,
        A_init_value = 1,
        A_upper = 1e-4,
        V_init_value = 1,
        V_lower = 1e-4,
    )
    return options
end