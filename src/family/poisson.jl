"""
    PoissonEPCA(indim::Integer, outdim::Integer; μ = 1, ϵ = eps())

Constructs an Exponential Family Principal Component Analysis (EPCA) model tailored for Poisson-distributed data.

The `PoissonEPCA` function creates an EPCA model designed to handle count data, assuming a Poisson distribution. This model minimizes the Bregman divergence based on the Poisson log-likelihood, making it suitable for datasets where each entry represents a non-negative integer count (e.g., event counts, word counts in documents).

# Arguments
- `indim::Integer`: The dimensionality of the input data (number of features). This represents the original high-dimensional space of the count data.

- `outdim::Integer`: The dimensionality of the output (reduced space). This is the target lower-dimensional representation the data will be compressed into.

# Keyword Arguments
- `μ::Real = 1`: The regularization parameter representing the expected mean of the Poisson distribution. Must be a positive value, as it corresponds to the mean of the distribution. Defaults to `1`.

- `ϵ::Real = eps()`: A small positive value added for numerical stability, especially to avoid logarithms of zero during the optimization process. Defaults to `eps()`, which is the smallest representable positive number such that `1.0 + eps()` is distinguishable from `1.0`.

# Returns
- `epca::EPCA`: An instance of the EPCA model initialized for Poisson-distributed data. This model can be fitted to count data using functions like `fit!`, and can then be used for data compression or reconstruction.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 50  # Input dimension (number of features)
outdim = 5  # Reduced output dimension (number of principal components)

# Create a Poisson EPCA model instance
poisson_epca_model = PoissonEPCA(indim, outdim; μ=1.0, ϵ=1e-6)

# Generate some random count data
X = rand(Poisson(1.0), 1000, indim)  # 1000 observations, each with 50 features (counts)

# Fit the model to the count data
A = fit!(poisson_epca_model, X; maxiter=200, verbose=true)

# The resulting matrix A is the lower-dimensional representation of X
```

# Notes
- The link function ``g(θ) = exp(θ)`` is the exponential function, which is appropriate for modeling Poisson-distributed data.
The function ``F(x)`` computes the Bregman divergence for the Poisson distribution, which is based on the Poisson log-likelihood.
The parameter `μ`` must be positive to ensure it is in the valid range for the mean of a Poisson distribution.
"""
function PoissonEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
)
    # assumes χ = ℕ
    F(x) = x * log(x + ϵ) - x
    g(θ) = exp(θ)
    @assert μ > 0 "For PoissonEPCA, μ must be positive to be in the range of g(θ) = exp(θ)."
    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g)); 
        μ = μ,
        ϵ = ϵ
    )
    return epca
end

