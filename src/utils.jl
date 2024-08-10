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

# TODO: convert this to a macro somehow + make the variable changable
function _symbolics_to_julia(symbolics_expression::Num)
    # NOTE: here symbolics_expression refers to the mathematical definition of expression, NOT a Julia Expr
    ex = quote
        θ -> $(Symbolics.toexpr(symbolics_expression))
    end
    eval(ex)
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

# TODO: rename these
# TODO double check math
function _invert_legrende(
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
    maxiter::Integer
) where T <: Real
    result = optimize(Â->L(Â * V), A)
    A = Optim.minimizer(result)
    if verbose && (i % steps_per_print == 0 || i == 1)
        loss = Optim.minimum(result)
        println("Iteration: $i/$maxiter | Loss: $loss")
    end
    return A
end

function _single_fit_iter(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    verbose::Bool,
    i::Integer,
    steps_per_print::Integer,
    maxiter::Integer
) where T <: Real
    V = Optim.minimizer(optimize(V̂->L(A * V̂), V))
    A = _single_compress_iter(
        L,
        V,
        A,
        verbose,
        i,
        steps_per_print,
        maxiter
    )
    return V, A
end

function _compress(
    L::Function,
    V::AbstractMatrix{T},
    A::AbstractMatrix{T},
    maxiter::Integer,
    verbose::Bool,
    steps_per_print::Integer,
) where T <: Real
    for i in 1:maxiter
        A = _single_compress_iter(
            L,
            V,
            A,
            verbose,
            i,
            steps_per_print,
            maxiter
        )
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
) where T <: Real
    for i in 1:maxiter
        V, A = _single_fit_iter(
            L,
            V,
            A,
            verbose,
            i,
            steps_per_print,
            maxiter
        )
    end
    return V, A
end

function _initialize_A(
    epca::EPCA,
    X::AbstractMatrix{T};
    A_init::Union{Nothing, AbstractMatrix{T}}=nothing
) where T <: Real
    n = size(X)[1]
    outdim = size(epca.V)[1]
    if isnothing(A_init)
        A = ones(n, outdim)
    else
        @assert size(A) == (n, outdim)
        A = A_init
    end
    return A
end