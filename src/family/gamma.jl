"""
    GammaEPCA(indim::Integer, outdim::Integer; μ = 1, ϵ = eps())

Constructs an Exponential Family Principal Component Analysis (EPCA) model tailored for Gamma-distributed data.

The `GammaEPCA` function creates an EPCA model designed to handle continuous positive data, assuming a Gamma distribution. This model minimizes the Bregman divergence specific to the Gamma distribution, making it suitable for datasets where each entry represents a positive continuous variable (e.g., financial data, durations).

# Arguments
- `indim::Integer`: The dimensionality of the input data (number of features). This represents the original high-dimensional space of the positive continuous data.

- `outdim::Integer`: The dimensionality of the output (reduced space). This is the target lower-dimensional representation the data will be compressed into.

# Keyword Arguments
- `μ::Real = 1`: The regularization parameter representing the mean parameter for the Gamma distribution. It must be nonzero as it influences the range of the link function `g(θ) = -1/θ`. Defaults to `1`.

- `ϵ::Real = eps()`: A small positive value added for numerical stability, particularly to avoid division by zero or logarithms of zero during the optimization process. Defaults to `eps()`, which is the smallest representable positive number such that `1.0 + eps()` is distinguishable from `1.0`.

# Returns
- `epca::EPCA`: An instance of the EPCA model initialized for Gamma-distributed data. This model can be fitted to positive continuous data using functions like `fit!`, and can then be used for data compression or reconstruction.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 50  # Input dimension (number of features)
outdim = 5  # Reduced output dimension (number of principal components)

# Create a Gamma EPCA model instance
gamma_epca_model = GammaEPCA(indim, outdim; μ=1.0, ϵ=1e-6)

# Generate some random positive continuous data
X = rand(Gamma(2.0, 1.0), 1000, indim)  # 1000 observations, each with 50 features (positive continuous values)

# Fit the model to the data
A = fit!(gamma_epca_model, X; maxiter=200, verbose=true)

# The resulting matrix A is the lower-dimensional representation of X
```

# Notes
- The Bregman divergence function ``Bg(x, θ) = -x * θ - log(-x * θ) - 1`` is specifically designed for the Gamma distribution.
- The link function ``g(θ) = -1/θ`` is appropriate for modeling Gamma-distributed data.
- The parameters for initialization are set to ensure that the product of matrices ``AV`` results in only negative entries, adhering to the domain requirements of the Gamma distribution.
"""
function GammaEPCA(indim::Integer, outdim::Integer)
    # χ = ℝ++
    Bg(x, θ) = -log(-x * θ) - x * θ
    g(θ) = -1 / θ
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = Options(
            A_init_value = -1,
            A_upper = -eps(),
            V_lower = eps()
        )
    )
    return epca
end

"""
Alias for [`GammaEPCA`](@ref).
"""
function ItakuraSaitoEPCA(indim::Integer, outdim::Integer)
    epca = GammaEPCA(indim, outdim)
    return epca
end