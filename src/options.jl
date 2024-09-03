@with_kw struct Options
    # symbolic calculus 
    metaprogramming::Bool = true

    # loss hyperparameters
    μ::Real = 1
    ϵ::Real = eps()

    A_init_value::Real = 1.0
    A_lower::Union{Real, Nothing} = nothing
    A_upper::Union{Real, Nothing} = nothing

    V_init_value::Real = 1.0
    V_lower::Union{Real, Nothing} = nothing
    V_upper::Union{Real, Nothing} = nothing

    # binary search options
    low = -1e10
    high = 1e10
    tol = 1e-10
    maxiter = 1e6
end