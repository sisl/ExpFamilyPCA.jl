"""
Supertype for exponential family principal component analysis models.
"""
abstract type EPCA end

"""
    fit!(epca::EPCA, X::AbstractMatrix{T}; maxiter::Integer = 100, verbose::Bool = false, steps_per_print::Integer = 10) where T <: Real

Fits the EPCA model on the dataset `X`. Call this function after creating an EPCA struct or to continue training on more data.

Should be called after creating an EPCA object or when you want to fit on new data.

!!! warning
    The default `fit!` may have long run times depending on the dataset size and model complexity. Consider adjusting the verbosity (`verbose`) and number of iterations (`maxiter`) to better balance runtime and model performance.

# Arguments
- `epca::EPCA`: The EPCA model. 
- `X::AbstractMatrix{T}`: (`n`, `indim`) - The input training data matrix. Rows are observations. Columns are features or variables. 

# Keyword Arguments
- `maxiter::Integer = 100`: The maximum number of iterations performed during loss minimization. Defaults to `100`. May converge early.
- `verbose::Bool = false`: A flag indicating whether to print optimization progress. If set to `true`, prints the loss value and iteration number at specified intervals (`steps_per_print`). Defaults to `false`.
- `steps_per_print::Integer = 10`: The number of iterations between printed progress updates when `verbose` is set to `true`. For example, if `steps_per_print` is `10`, progress will be printed every 10 iterations. Defaults to `10`.

# Returns
- `A::AbstractMatrix{T}`: (`n`, `outdim`) - The compressed data.

# Usage

**Input:**
```julia
using ExpFamilyPCA
using Random; Random.seed!(1)

# Create the model
indim = 10  # Input dimension
outdim = 5  # Output dimension
epca = BernoulliEPCA(indim, outdim)

# Generate some random training data
n = 100
X = rand(0:1, n, indim)

# Fit the model to the data
A = fit!(epca, X; maxiter=200, verbose=true, steps_per_print=50);
```

**Output:**
```@repl
Iteration: 1/200 | Loss: 31.7721864082419
Iteration: 50/200 | Loss: 11.07389383509631
Iteration: 100/200 | Loss: 10.971490262772905
Iteration: 150/200 | Loss: 10.886018474442618
Iteration: 200/200 | Loss: 10.718703556787007
```
"""
function fit! end

"""
    compress(epca::EPCA, X::AbstractMatrix{T}; maxiter::Integer = 100, verbose::Bool = false, steps_per_print::Integer = 10) where T <: Real

Compresses the input data `X` with the EPCA model.

# Arguments
- `epca::EPCA`: The fitted EPCA model.[^1] `fit!` should be called before `compress`.
- `X::AbstractMatrix{T}`: (`n`, `indim`) - The input data matrix (can differ from the training data). Rows are observations. Columns are features or variables. 

# Keyword Arguments
- `maxiter::Integer = 100`: The maximum number of iterations performed during loss minimization. Defaults to `100`. May converge early.
- `verbose::Bool = false`: A flag indicating whether to print optimization progress. If set to `true`, prints the loss value and iteration number at specified intervals (`steps_per_print`). Defaults to `false`.
- `steps_per_print::Integer = 10`: The number of iterations between printed progress updates when `verbose` is set to `true`. For example, if `steps_per_print` is `10`, progress will be printed every 10 iterations. Defaults to `10`.

# Returns
- `A::AbstractMatrix{T}`: (`n`, `outdim`) - The compressed data.

# Usage
```julia
# Generate some random test data
m = 10
Y = rand(0:1, m, indim)

# Compress the test data using the fitted model from the previous example
Y_compressed = compress(epca, Y)
```

[^1]: If `compress` is called before `fit!`, `X` will compressed using unfitted starting weights.
"""
function compress end

"""
    decompress(epca::EPCA, A::AbstractMatrix{T}) where T <: Real

Decompress the compressed matrix `A` with the EPCA model.

# Arguments
- `epca::EPCA`: The fitted EPCA model.[^1] `fit!` should be called before `compress`.
- `A::AbstractMatrix{T}`: (`n`, `outdim`) - A compressed data matrix.

# Returns
- `X̂::AbstractMatrix{T}`: (`n`, `indim`) - The reconstructed data matrix approximated using EPCA model parameters.

# Usage
```julia
Y_reconstructed = decompress(epca, Y)
```
"""
function decompress end

function fit!(
    epca::EPCA,
    X::AbstractMatrix{<:Real};
    maxiter::Integer = 100,
    verbose::Bool = false,
    steps_per_print::Integer = 10,
)
    L = _make_loss(epca, X)
    A = _initialize_A(epca, X)

    V, A = _fit(
        L,
        epca.V,
        A,
        X,
        maxiter,
        verbose,
        steps_per_print,
        epca.options
    )
    epca.V[:] = V
    return A
end

function compress(
    epca::EPCA,
    X::AbstractMatrix{<:Real};
    maxiter::Integer = 100,
    verbose::Bool = false,
    steps_per_print::Integer = 10
)
    L = _make_loss(epca, X)
    A = _initialize_A(epca, X)
    A = _compress(
        L,
        epca.V,
        A,
        X,
        maxiter,
        verbose,
        steps_per_print,
        epca.options
    )
    return A
end

function decompress(
    epca::EPCA,
    A::AbstractMatrix{<:Real}
)
    Θ = A * epca.V
    X̂ = epca.g.(Θ)
    return X̂
end
