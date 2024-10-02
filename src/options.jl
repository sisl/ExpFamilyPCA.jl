"""
    Options(; metaprogramming::Bool = true, μ::Real = 1, ϵ::Real = eps(), A_init_value::Real = 1.0, A_lower::Union{Real, Nothing} = nothing, A_upper::Union{Real, Nothing} = nothing, A_use_sobol::Bool = false, V_init_value::Real = 1.0, V_lower::Union{Real, Nothing} = nothing, V_upper::Union{Real, Nothing} = nothing, V_use_sobol::Bool = false, low = -1e10, high = 1e10, tol = 1e-10, maxiter = 1e6)

Defines a struct `Options` for configuring various parameters used in optimization and calculus. It provides flexible defaults for metaprogramming, initialization values, optimization boundaries, and binary search controls.

# Fields
- `metaprogramming::Bool`: Enables metaprogramming for symbolic calculus conversions. Default is `true`.
- `μ::Real`: A regularization hyperparameter. Default is `1`.
- `ϵ::Real`: A regularization hyperparameter. Default is `eps()`.
- `A_init_value::Real`: Initial value for parameter `A`. Default is `1.0`.
- `A_lower::Union{Real, Nothing}`: Lower bound for `A`, or `nothing`. Default is `nothing`.
- `A_upper::Union{Real, Nothing}`: Upper bound for `A`, or `nothing`. Default is `nothing`.
- `A_use_sobol::Bool`: Use Sobol sequences for initializing `A`. Default is `false`.
- `V_init_value::Real`: Initial value for parameter `V`. Default is `1.0`.
- `V_lower::Union{Real, Nothing}`: Lower bound for `V`, or `nothing`. Default is `nothing`.
- `V_upper::Union{Real, Nothing}`: Upper bound for `V`, or `nothing`. Default is `nothing`.
- `V_use_sobol::Bool`: Use Sobol sequences for initializing `V`. Default is `false`.
- `low::Real`: Lower bound for binary search. Default is `-1e10`.
- `high::Real`: Upper bound for binary search. Default is `1e10`.
- `tol::Real`: Tolerance for stopping binary search. Default is `1e-10`.
- `maxiter::Real`: Maximum iterations for binary search. Default is `1e6`.
"""
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

"""
    NegativeDomain(; metaprogramming::Bool = true, μ::Real = 1, ϵ::Real = eps(), low::Real = -1e10, high::Real = 1e10, tol::Real = 1e-10, maxiter::Real = 1e6)

Returns an instance of `Options` configured for optimization over the negative domain. Sets defaults for `A` and `V` parameters while keeping the remaining settings from `Options`.

# Specific Settings
- `A_init_value = -1`: Initializes `A` with a negative value.
- `A_upper = -1e-4`: Upper bound for `A` is constrained to a small negative value.
- `V_init_value = 1`: Initializes `V` with a positive value.
- `V_lower = 1e-4`: Lower bound for `V` is constrained to a small positive value.

Other fields inherit from the `Options` struct.
"""
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

"""
    PositiveDomain(; metaprogramming::Bool = true, μ::Real = 1, ϵ::Real = eps(), low::Real = -1e10, high::Real = 1e10, tol::Real = 1e-10, maxiter::Real = 1e6)

Returns an instance of `Options` configured for optimization over the positive domain. Sets defaults for `A` and `V` parameters while keeping the remaining settings from `Options`.

# Specific Settings
- `A_init_value = 1`: Initializes `A` with a positive value.
- `A_upper = 1e-4`: Upper bound for `A` is constrained to a small positive value.
- `V_init_value = 1`: Initializes `V` with a positive value.
- `V_lower = 1e-4`: Lower bound for `V` is constrained to a small positive value.

Other fields inherit from the `Options` struct.
"""
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