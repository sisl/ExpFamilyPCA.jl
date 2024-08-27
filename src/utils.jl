function _symbolics_to_julia(
    symbolics_expression::Num,
    symbolics_variable::Num;
)
    # NOTE: here symbolics_expression refers to the mathematical definition of expression, NOT a Julia Expr
    # NOTE: make sure that derivatives are expanded with expand_derivatives before calling the helper
    fn = x->begin
        result = substitute(
            symbolics_expression, 
            Dict(symbolics_variable => x)
        ).val
        return result
    end
    return fn
end

function _symbolics_to_julia(symbolics_expression::Num)
    # NOTE: here symbolics_expression refers to the mathematical definition of expression, NOT a Julia Expr
    ex = quote
        θ -> $(Symbolics.toexpr(symbolics_expression))
    end
    eval(ex) |> FunctionWrapper{Float64, Tuple{Float64}}
end

function _binary_search_monotone(
    f, 
    target; 
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)
    iter = 0
    while high - low > tol && iter < maxiter
        mid = (low + high) / 2
        if f(mid) < target
            low = mid
        else
            high = mid
        end
        iter += 1
    end
    return (low + high) / 2
end

"""Invert Legendre transformation"""
function _invert_legendre(
    f;
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)
    g(x) = _binary_search_monotone(
        f, 
        x; 
        low=low,
        high=high,
        tol=tol,
        maxiter=maxiter
    )
    return g
end

function _single_compress_iter(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    verbose::Bool,
    i::Integer,
    steps_per_print::Integer,
    maxiter::Integer;
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
) where T <: Real
    if isnothing(A_lower) && isnothing(A_upper)
        result = optimize(Â->L(Â * V), A)
    elseif isnothing(A_lower) && !isnothing(A_upper)
        result = optimize(Â->L(Â * V), -Inf, A_upper, A)
    elseif !isnothing(A_lower) && isnothing(A_upper)
        result = optimize(Â->L(Â * V), A_lower, Inf, A)
    else
        @assert A_lower <= A_upper "A_lower must be <= A_upper"
        result = optimize(Â->L(Â * V), A_lower, A_upper, A)
    end

    A = Optim.minimizer(result)
    loss = Optim.minimum(result)
    if verbose && (i % steps_per_print == 0 || i == 1)
        println("Iteration: $i/$maxiter | Loss: $loss")
    end
    return A, loss
end

function _single_fit_iter(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    verbose::Bool,
    i::Integer,
    steps_per_print::Integer,
    maxiter::Integer;
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
) where T <: Real
    if isnothing(V_lower) && isnothing(V_upper)
        result = optimize(V̂->L(A * V̂), V)
    elseif isnothing(V_lower) && !isnothing(V_upper)
        result = optimize(V̂->L(A * V̂), -Inf, V_upper, V)
    elseif !isnothing(V_lower) && isnothing(V_upper)
        result = optimize(V̂->L(A * V̂), V_lower, Inf, V)
    else
        @assert V_lower <= V_upper "V_lower must be <= V_upper"
        result = optimize(V̂->L(A * V̂), V_lower, V_upper, V)
    end

    V = Optim.minimizer(result)
    A, loss = _single_compress_iter(
        L,
        V,
        A,
        verbose,
        i,
        steps_per_print,
        maxiter;
        A_lower = A_lower,
        A_upper = A_upper,
    )
    return V, A, loss
end

function _compress(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    maxiter::Integer,
    verbose::Bool,
    steps_per_print::Integer;
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing
) where T <: Real
    for i in 1:maxiter
        A, loss = _single_compress_iter(
            L,
            V,
            A,
            verbose,
            i,
            steps_per_print,
            maxiter;
            A_lower = A_lower,
            A_upper = A_upper,
        )
        if isnan(loss)
            @warn "Loss is NaN, ending early at iteration $i."
            break
        end
        if !isfinite(loss)
            @warn "Loss not finite, ending early at iteration $i."
            break
        end
    end
    return A
end

function _fit(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    maxiter::Integer,
    verbose::Bool,
    steps_per_print::Integer;
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
) where T <: Real
    for i in 1:maxiter
        V, A, loss = _single_fit_iter(
            L,
            V,
            A,
            verbose,
            i,
            steps_per_print,
            maxiter;
            A_lower = A_lower,
            A_upper = A_upper,
            V_lower = V_lower,
            V_upper = V_upper
        )
        if isnan(loss)
            @warn "Loss is NaN, ending early at iteration $i."
            break
        end
        if !isfinite(loss)
            @warn "Loss not finite, ending early at iteration $i."
        end
    end
    return V, A
end

function _initialize_A(
    epca::EPCA,
    X::AbstractMatrix{<:Real}
)
    T = eltype(epca.V)
    n = size(X)[1]
    outdim = size(epca.V)[1]
    A_init_value = epca.A_init_value
    if isnothing(A_init_value)
        A = ones(T, n, outdim)
    else
        A = fill(T(A_init_value), n, outdim)
    end
    return A
end
