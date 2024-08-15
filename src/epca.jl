"""
    EPCA

## Constructors

### EPCA 1
    EPCA(indim::Integer, outdim::Integer, g::Function, F::Function, ::Val{(:F, :g)}; μ=1, ϵ=eps())
    EPCA(indim::Integer, outdim::Integer, F::Function, f::Function, ::Val{(:F, :f)}; μ=1, ϵ=eps(), low=-1e10, high=1e10, tol=1e-10, maxiter=1e6)
    EPCA(indim::Integer, outdim::Integer, F::Function, ::Val{(:F)}; μ=1, ϵ=eps(), metaprogramming=true, low=-1e10, high=1e10, tol=1e-10, maxiter=1e6)
    EPCA(indim::Integer, outdim::Integer, F::Function, G::Function, ::Val{(:F, :G)}; μ=1, ϵ=eps(), metaprogramming=true)

### EPCA 2
    EPCA(indim::Integer, outdim::Integer, G::Function, g::Function, ::Val{(:G, :g)}; tol=eps(), μ=1, ϵ=eps(), metaprogramming=true)
    EPCA(indim::Integer, outdim::Integer, G::Function, ::Val{(:G)}; tol=eps(), μ=1, ϵ=eps(), metaprogramming=true)

### EPCA 3
    EPCA(indim::Integer, outdim::Integer, Bregman::Union{Function, PreMetric}, g::Function, ::Val{(:Bregman, :g)}; μ=1, ϵ=eps())
    EPCA(indim::Integer, outdim::Integer, Bregman::Function, G::Function, ::Val{(:Bregman, :G)}; μ=1, ϵ=eps(), metaprogramming=true)

"""
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
    maxiter::Integer=10,
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