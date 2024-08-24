"""
    EPCA

A flexible type representing an Exponential Family Principal Component Analysis (EPCA) model. There are many ways equivalent ways specify an EPCA model, so we provide several construction signatures. All constructions are theoretically equivalent, though some constructors are more stable and performant.

The `EPCA` method (and abstract type) wraps several unexported subtypes `EPCA1`, `EPCA2` and `EPCA3`. Their fields differ slightly, but they share a common interface which is documented below.

# Fields
- `V::AbstractMatrix{<:Real}`: Internal matrix used by the model.
- `g::Function`: The link function applied to natural parameters during the fitting process.
- `μ::Real`: A hyperparameter representing a positive number, used in regularization.
- `ϵ::Real`: A hyperparameter representing a nonnegative number, used in regularization.

# Constructors
### EPCA 1
- `EPCA(indim::Integer, outdim::Integer, F::Function, g::Function, ::Val{(:F, :g)}; μ=1, ϵ=eps())`
- `EPCA(indim::Integer, outdim::Integer, F::Function, f::Function, ::Val{(:F, :f)}; μ=1, ϵ=eps(), low=-1e10, high=1e10, tol=1e-10, maxiter=1e6)`
- `EPCA(indim::Integer, outdim::Integer, F::Function, ::Val{(:F)}; μ=1, ϵ=eps(), metaprogramming=true, low=-1e10, high=1e10, tol=1e-10, maxiter=1e6)`
- `EPCA(indim::Integer, outdim::Integer, F::Function, G::Function, ::Val{(:F, :G)}; μ=1, ϵ=eps(), metaprogramming=true)`

### EPCA 2
- `EPCA(indim::Integer, outdim::Integer, G::Function, g::Function, ::Val{(:G, :g)}; tol=eps(), μ=1, ϵ=eps())`
- `EPCA(indim::Integer, outdim::Integer, G::Function, ::Val{(:G)}; tol=eps(), μ=1, ϵ=eps(), metaprogramming=true)`

### EPCA 3
- `EPCA(indim::Integer, outdim::Integer, Bregman::Union{Function, PreMetric}, g::Function, ::Val{(:Bregman, :g)}; μ=1, ϵ=eps())`
- `EPCA(indim::Integer, outdim::Integer, Bregman::Function, G::Function, ::Val{(:Bregman, :G)}; μ=1, ϵ=eps(), metaprogramming=true)`

### EPCA 4


# Notes
- The `EPCA` type enforces constraints such as `indim >= outdim` to ensure the input dimension is not smaller than the output dimension.
- Depending on the constructor used, different forms of the loss function and internal transformations will be employed during model fitting.
"""
abstract type EPCA end


"""
    fit!(epca::EPCA, X::AbstractMatrix{T}; maxiter::Integer=10, verbose::Bool=false, steps_per_print::Integer=10, A_init::Union{Nothing, AbstractMatrix{T}}=nothing, autodiff::Bool=false) where T <: Real

Fits the EPCA model to the data matrix `X`.

# Arguments
- `epca::EPCA`: The EPCA model to be fitted.
- `X::AbstractMatrix{T}`: The data matrix on which the model is to be fitted, where `T` is a subtype of `Real`.
- `maxiter::Integer=10`: The maximum number of iterations for the fitting process (default is 10).
- `verbose::Bool=false`: If `true`, detailed progress is printed during the fitting process (default is `false`).
- `steps_per_print::Integer=10`: The number of steps between each progress printout if `verbose` is `true` (default is every 10 steps).
- `A_init::Union{Nothing, AbstractMatrix{T}}=nothing`: An optional initial value for the matrix `A`. If not provided, the matrix will be initialized automatically.
- `autodiff::Bool=false`: If `true`, automatic differentiation is used during the fitting process (default is `false`).

# Returns
- `A::AbstractMatrix{T}`: The matrix `A` resulting from the fitting process.

The function updates the `V` parameter of the `EPCA` model and returns the matrix `A` as the result of the fitting process.
"""
function fit! end

"""
    compress(epca::EPCA, X::AbstractMatrix{T}; maxiter::Integer=10, verbose::Bool=false, steps_per_print::Integer=10, A_init::Union{Nothing, AbstractMatrix{T}}=nothing, autodiff::Bool=false) where T <: Real

Compresses the data matrix `X` using the EPCA model.

# Arguments
- `epca::EPCA`: The EPCA model used for compressing the data.
- `X::AbstractMatrix{T}`: The data matrix to be compressed, where `T` is a subtype of `Real`.
- `maxiter::Integer=10`: The maximum number of iterations for the compression process (default is 10).
- `verbose::Bool=false`: If `true`, detailed progress is printed during the compression process (default is `false`).
- `steps_per_print::Integer=10`: The number of steps between each progress printout if `verbose` is `true` (default is every 10 steps).
- `A_init::Union{Nothing, AbstractMatrix{T}}=nothing`: An optional initial value for the matrix `A`. If not provided, the matrix will be initialized automatically.
- `autodiff::Bool=false`: If `true`, automatic differentiation is used during the compression process (default is `false`).

# Returns
- `A::AbstractMatrix{T}`: The matrix `A` representing the compressed version of the data matrix `X`.

The function performs compression based on the EPCA model and returns the compressed matrix `A`.
"""
function compress end

"""
    decompress(epca::EPCA, A::AbstractMatrix{T}) where T <: Real

Decompresses the matrix `A` using the EPCA model to reconstruct the original data matrix.

# Arguments
- `epca::EPCA`: The EPCA model used for decompressing the data.
- `A::AbstractMatrix{T}`: The compressed matrix to be decompressed, where `T` is a subtype of `Real`.

# Returns
- `X̂::AbstractMatrix{T}`: The reconstructed data matrix, which approximates the original data matrix before compression.

The function applies the inverse transformation defined by the `EPCA` model to decompress `A` and recover an approximation of the original data matrix.
"""
function decompress end

function fit!(
    epca::EPCA,
    X::AbstractMatrix{T};
    maxiter::Integer=100,
    verbose::Bool=false,
    steps_per_print::Integer=10,
    A_init::Union{Nothing, AbstractMatrix{T}}=nothing,
    autodiff::Bool=false
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
        steps_per_print,
        autodiff
    )
    epca.V[:] = V
    return A
end

function compress(
    epca::EPCA,
    X::AbstractMatrix{T};
    maxiter::Integer=100,
    verbose::Bool=false,
    steps_per_print::Integer=10,
    A_init::Union{Nothing, AbstractMatrix{T}}=nothing,
    autodiff::Bool=false
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
        steps_per_print,
        autodiff,
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