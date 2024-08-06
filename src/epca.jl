abstract type EPCA{T <: Real} end

function fit! end
function compress end
function decompress end

### UTILITIES ### 

function _make_loss end

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
    epca::EPCA{T},
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

### BODY ####

function fit!(
    epca::EPCA{T},
    X::AbstractMatrix{T};
    maxiter::Integer=10,
    verbose::Bool=false,
    steps_per_print::Integer=10,
    A_init::Union{Nothing, AbstractMatrix{T}}=nothing
) where T <: Real
    L = _make_loss(epca, X)
    V = epca.V
    A = _initialize_A(
        epca,
        X;
        A_init=A_init
    )
    V, A = _fit(
        L,
        V,
        A,
        maxiter,
        verbose,
        steps_per_print
    )
    epca.V[:] = V
    return A
end

function compress(
    epca::EPCA{T},
    X::AbstractMatrix{T};
    maxiter::Integer=10,
    verbose::Bool=false,
    steps_per_print::Integer=10,
    A_init::Union{Nothing, AbstractMatrix{T}}=nothing
) where T <: Real
    L = _make_loss(epca, X)
    V = epca.V
    A = _initialize_A(
        epca,
        X;
        A_init=A_init
    )
    A = _compress(
        L,
        V,
        A,
        maxiter,
        verbose,
        steps_per_print
    )
    return A
end

function decompress(
    epca::EPCA{T},
    A::AbstractMatrix{T}
) where T <: Real
    natural_params = A * epca.V  # irritatingly, VSCode's default font cascade doesn't distinguish between uppercase Θ and lowercase θ
    X̂ = epca.g.(natural_params)
    return X̂
end