abstract type EPCA end

function fit! end
function compress end
function decompress end

function fit!(
    epca::EPCA,
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
    epca::EPCA,
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
    epca::EPCA,
    A::AbstractMatrix{T}
) where T <: Real
    natural_params = A * epca.V  # irritatingly, VSCode's default font cascade doesn't distinguish between uppercase Θ and lowercase θ
    X̂ = epca.g.(natural_params)
    return X̂
end