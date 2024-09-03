abstract type EPCA end


"""
    fit!(epca::EPCA, X::AbstractMatrix{T}; maxiter::Integer = 100, verbose::Bool = false, steps_per_print::Integer = 10) where T <: Real

Fits the Exponential Family Principal Component Analysis (EPCA) model to the given dataset `X`.

The `fit!` function optimizes the parameters of an EPCA model to minimize a loss function specific to the chosen exponential family distribution of the data. This optimization process adjusts the model's internal parameters to achieve a lower-dimensional representation of the data that captures the most significant variance while adhering to the constraints of the distribution family.

# Arguments
- `epca::EPCA`: An instance of the EPCA model. This object specifies the structure of the model and the assumptions about the distribution of the input data. It must be an instance of a subtype of the abstract type `EPCA`.

- `X::AbstractMatrix{T}`: The input data matrix where each row represents an observation and each column represents a feature or variable. `T` is a subtype of `Real`, indicating that the data should consist of real numbers (e.g., `Float64`, `Float32`).

# Keyword Arguments
- `maxiter::Integer = 100`: The maximum number of iterations to perform during the optimization process. Each iteration updates the model parameters to reduce the loss function. Defaults to `100`.

- `verbose::Bool = false`: A flag indicating whether to print progress information during the optimization process. If set to `true`, the function prints the loss value and iteration number at specified intervals (`steps_per_print`). Defaults to `false`.

- `steps_per_print::Integer = 10`: The number of iterations between printed progress updates when `verbose` is set to `true`. For example, if `steps_per_print` is `10`, progress will be printed every 10 iterations. Defaults to `10`.

# Returns
- `A::AbstractMatrix{T}`: The optimized auxiliary parameter matrix `A` that minimizes the loss function for the model. This matrix represents the lower-dimensional representation of the input data `X` in the reduced space defined by the EPCA model. The matrix `A` can be used for further data compression and reconstruction tasks.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 100  # Input dimension
outdim = 10  # Reduced output dimension
G = x -> log(1 + exp(x))  # Log-partition function for Bernoulli data
g = x -> 1 / (1 + exp(-x))  # Sigmoid link function

# Create an EPCA model instance
epca_model = EPCA(indim, outdim, G, g, Val((:G, :g)))

# Generate some random input data
X = rand(Float64, 1000, indim)  # 1000 observations, each with 100 features

# Fit the model to the data
A = fit!(epca_model, X; maxiter=200, verbose=true, steps_per_print=20)

# The resulting matrix A is the lower-dimensional representation of X
```

# Notes
- It is recommended to preprocess the input data X (e.g., normalization or scaling) to ensure better convergence and numerical stability during optimization.
"""
function fit! end

"""
    compress(epca::EPCA, X::AbstractMatrix{T}; maxiter::Integer = 100, verbose::Bool = false, steps_per_print::Integer = 10) where T <: Real

Compresses the input data `X` using a fitted Exponential Family Principal Component Analysis (EPCA) model.

The `compress` function projects the input data `X` into a lower-dimensional space defined by the fitted EPCA model. This compression process reduces the dimensionality of the data while preserving its most significant features, as dictated by the model's parameters and the underlying exponential family distribution.

# Arguments
- `epca::EPCA`: A pre-fitted instance of the EPCA model. This object should have its parameters already optimized by a prior call to `fit!`, specifying the structure of the model and the distributional assumptions of the data.

- `X::AbstractMatrix{T}`: The input data matrix to be compressed, where each row represents an observation and each column represents a feature. `T` must be a subtype of `Real`, indicating the data consists of real numbers (e.g., `Float64`, `Float32`).

# Keyword Arguments
- `maxiter::Integer = 100`: The maximum number of iterations for the optimization process to compress the data. Each iteration updates the auxiliary parameter matrix `A` to best represent the data in the reduced space. Defaults to `100`.

- `verbose::Bool = false`: A flag indicating whether to print progress information during the compression process. If set to `true`, the function prints the loss value and iteration number at specified intervals (`steps_per_print`). Defaults to `false`.

- `steps_per_print::Integer = 10`: The number of iterations between printed progress updates when `verbose` is set to `true`. For example, if `steps_per_print` is `10`, progress will be printed every 10 iterations. Defaults to `10`.

# Returns
- `A::AbstractMatrix{T}`: The compressed representation of the input data `X`. This matrix `A` represents the input data in the lower-dimensional space defined by the EPCA model, retaining the essential structure and patterns of the data with reduced dimensionality.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 100  # Input dimension
outdim = 10  # Reduced output dimension
G = x -> log(1 + exp(x))  # Log-partition function for Bernoulli data
g = x -> 1 / (1 + exp(-x))  # Sigmoid link function

# Create and fit an EPCA model instance
epca_model = EPCA(indim, outdim, G, g, Val((:G, :g)))
X = rand(Float64, 1000, indim)  # Generate some random input data
fit!(epca_model, X; maxiter=200, verbose=true)

# Compress the data using the fitted model
A_compressed = compress(epca_model, X; maxiter=100, verbose=true, steps_per_print=20)

# The resulting matrix A_compressed is the lower-dimensional representation of X
```
"""
function compress end

"""
    decompress(epca::EPCA, A::AbstractMatrix{T}) where T <: Real

Reconstructs the original data from its compressed form using a fitted Exponential Family Principal Component Analysis (EPCA) model.

The `decompress` function takes a compressed representation `A` of the data and reconstructs an approximation of the original data in its full-dimensional space. This reconstruction is based on the learned parameters of the EPCA model, which captures the underlying patterns and structure of the original dataset.

# Arguments
- `epca::EPCA`: A fitted instance of the EPCA model. This object should have been optimized with a prior call to `fit!` and should reflect the structure and distributional assumptions of the original data.

- `A::AbstractMatrix{T}`: The compressed data matrix, where each row represents a lower-dimensional observation and each column represents a reduced feature. `T` must be a subtype of `Real`, indicating the data consists of real numbers (e.g., `Float64`, `Float32`).

# Returns
- `X̂::AbstractMatrix{T}`: The reconstructed data matrix in its original dimensionality. This matrix `X̂` is an approximation of the original input data `X`, reconstructed using the parameters of the EPCA model.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 100  # Input dimension
outdim = 10  # Reduced output dimension
G = x -> log(1 + exp(x))  # Log-partition function for Bernoulli data
g = x -> 1 / (1 + exp(-x))  # Sigmoid link function

# Create and fit an EPCA model instance
epca_model = EPCA(indim, outdim, G, g, Val((:G, :g)))
X = rand(Float64, 1000, indim)  # Generate some random input data
fit!(epca_model, X; maxiter=200, verbose=true)

# Compress the data using the fitted model
A_compressed = compress(epca_model, X; maxiter=100, verbose=true, steps_per_print=20)

# Decompress the data to approximate the original data
X_reconstructed = decompress(epca_model, A_compressed)

# The resulting matrix X_reconstructed is an approximation of the original data X
```

# Notes
- The decompression process uses the link function `g`` defined in the EPCA model to transform the natural parameters back into the mean parameters of the distribution.
- This function assumes that the input `A`` is a valid compressed representation obtained via the compress function with the same EPCA model.
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
    natural_params = A * epca.V
    X̂ = epca.g.(natural_params)
    return X̂
end
