"""
    NormalEPCA(indim::Integer, outdim::Integer; μ = 1, ϵ = eps())

Constructs an Exponential Family Principal Component Analysis (EPCA) model tailored for normally-distributed (Gaussian) data.

The `NormalEPCA` function creates an EPCA model designed for continuous data that follows a normal distribution. This model is equivalent to the generic Principal Component Analysis (PCA), as it minimizes the squared Euclidean distance, which is appropriate for Gaussian-distributed data.

# Arguments
- `indim::Integer`: The dimensionality of the input data (number of features). This represents the original high-dimensional space of the normally-distributed data.

- `outdim::Integer`: The dimensionality of the output (reduced space). This is the target lower-dimensional representation the data will be compressed into.

# Keyword Arguments
- `μ::Real = 1`: The regularization parameter representing the mean of the normal distribution. Defaults to `1`.

- `ϵ::Real = eps()`: A small positive value added for numerical stability, particularly to avoid issues during the optimization process. Defaults to `eps()`, which is the smallest representable positive number such that `1.0 + eps()` is distinguishable from `1.0`.

# Returns
- `epca::EPCA`: An instance of the EPCA model initialized for normally-distributed data. This model can be fitted to Gaussian data using functions like `fit!`, and can then be used for data compression or reconstruction.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 100  # Input dimension (number of features)
outdim = 10  # Reduced output dimension (number of principal components)

# Create a Normal EPCA model instance
normal_epca_model = NormalEPCA(indim, outdim; μ=1.0, ϵ=1e-6)

# Generate some random normally-distributed data
X = randn(1000, indim)  # 1000 observations, each with 100 features (Gaussian)

# Fit the model to the data
A = fit!(normal_epca_model, X; maxiter=200, verbose=true)

# The resulting matrix A is the lower-dimensional representation of X
```

# Notes
- The Bregman divergence function ``B_F(p, q) = (p-q)^2 / 2`` computes half of the squared Euclidean distance, which is equivalent to the loss function minimized in PCA.
- The link function ``g`` is the identity function, making this model equivalent to classical PCA for Gaussian data.
- The NormalEPCA function is synonymous with generic PCA but formulated within the EPCA framework.
"""
function NormalEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
)
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    B(p, q) = Distances.sqeuclidean(p, q) / 2
    g = identity
    epca = EPCA(
        indim,
        outdim,
        B,
        g,
        Val((:B, :g));
        μ = μ,
        ϵ = ϵ
    )
    return epca
end

"""
Alias for [`NormalEPCA`](@ref).
"""
function GaussianEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
)
    epca = NormalEPCA(
        indim,
        outdim;
        μ = μ,
        ϵ = ϵ
    )
    return epca
end