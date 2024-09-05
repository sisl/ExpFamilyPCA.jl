function _check_dimensions(
    indim::Integer, 
    outdim::Integer
)
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
end

function _check_binary_search_arguments(
    low::Real,
    high::Real,
    tol::Real,
    maxiter::Real
)
    @assert low < high "Low bound (low) must be less than high bound (high)."
    @assert tol > 0 "Tolerance (tol) must be a positive number."
    @assert maxiter > 0 "Maximum iterations (maxiter) must be a positive number."   
end

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
function _invert_legendre(f, options::Options)
    @unpack low, high, tol, maxiter = options
    _check_binary_search_arguments(low, high, tol, maxiter)
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

function is_constant_matrix(A::AbstractMatrix)
    flag = length(unique(A)) == 1
    return flag
end

function _optimize(
    f::Function, 
    lower::Union{Real, Nothing}, 
    upper::Union{Real, Nothing}, 
    X0::AbstractMatrix{T}
) where T <: Real
    if isnothing(lower) && isnothing(upper)
        result = optimize(f, X0)
    elseif isnothing(lower) && !isnothing(upper)
        result = optimize(f, -Inf, upper, X0)
    elseif !isnothing(lower) && isnothing(upper)
        result = optimize(f, lower, Inf, X0)
    else
        result = optimize(f, lower, upper, X0)
    end

    minimizer = Optim.minimizer(result)
    loss = Optim.minimum(result)

    return minimizer, loss
end

function _single_compress_iter(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    verbose::Bool,
    i::Integer,
    steps_per_print::Integer,
    maxiter::Integer,
    options::Options
) where T <: Real
    @unpack A_lower, A_upper = options
    A, loss = _optimize(
        Â->L(Â * V), 
        A_lower,
        A_upper,
        A
    )
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
    maxiter::Integer,
    options::Options
) where T <: Real
    @unpack V_lower, V_upper = options
    V, _ = _optimize(
        V̂->L(A * V̂), 
        V_lower,
        V_upper,
        V
    )
    A, loss = _single_compress_iter(
        L,
        V,
        A,
        verbose,
        i,
        steps_per_print,
        maxiter,
        options
    )
    return V, A, loss
end

function _check_convergence(loss, last_loss; verbose=false)    
    # Check if loss has converged
    if !ismissing(last_loss) && last_loss == loss
        if verbose
            println("Loss converged early. Stopping iteration.")
        end
        return true  # Indicates that the iteration should stop
    end
        
    # Check for NaN loss
    if isnan(loss)
        @warn "Loss is NaN. Stopping iteration."
        return true
    end
    
    # Check for non-finite loss
    if !isfinite(loss)
        @warn "Loss diverged. Stopping iteration."
        return true
    end
    
    return false
end

function _compress(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    maxiter::Integer,
    verbose::Bool,
    steps_per_print::Integer,
    options::Options
) where T <: Real
    last_loss = missing
    for i in 1:maxiter
        A, loss = _single_compress_iter(
            L,
            V,
            A,
            verbose,
            i,
            steps_per_print,
            maxiter,
            options
        )
        if _check_convergence(loss, last_loss; verbose=verbose)
            break
        end
        last_loss = loss
    end
    return A
end

function _fit(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    maxiter::Integer,
    verbose::Bool,
    steps_per_print::Integer,
    options::Options
) where T <: Real
    last_loss = missing
    for i in 1:maxiter
        V, A, loss = _single_fit_iter(
            L,
            V,
            A,
            verbose,
            i,
            steps_per_print,
            maxiter,
            options
        )
        if _check_convergence(loss, last_loss; verbose=verbose)
            break
        end
        last_loss = loss
    end
    return V, A
end

function _make_sobol_matrix(m, n)
    # TODO: add support for arbitrary hypercube rescaling
    s = SobolSeq(n)
    result = reduce(vcat, [next!(s) for _ = 1:m]')
    return result
end

function _initialize_A(epca::EPCA, X::AbstractMatrix{<:Real})
    V = epca.V
    n = size(X)[1]
    outdim = size(V)[1]
    if epca.options.A_use_sobol
        A = _make_sobol_matrix(n, outdim)
    else
        A_init_value = epca.options.A_init_value
        T = eltype(V)
        A = fill(T(A_init_value), n, outdim)
    end
    return A
end

function _initialize_V(indim::Integer, outdim::Integer, options::Options)
    if options.V_use_sobol
        V = _make_sobol_matrix(outdim, indim)
    else
        V = fill(Float64(options.V_init_value), outdim, indim)
    end
    return V
end

function _differentiate(H, metaprogramming::Bool)
    @variables θ
    D = Differential(θ)
    _h = expand_derivatives(D(H(θ)))
    if metaprogramming
        h = _symbolics_to_julia(_h)
    else
        h = _symbolics_to_julia(_h, θ)
    end
    return h
end