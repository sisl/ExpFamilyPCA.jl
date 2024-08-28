
"""
    BernoulliEPCA(indim::Integer, outdim::Integer; μ = 0.5, ϵ = eps())

Constructs an Exponential Family Principal Component Analysis (EPCA) model tailored for Bernoulli-distributed data.

The `BernoulliEPCA` function creates an EPCA model that is specifically designed to handle binary data, assuming a Bernoulli distribution. The model is based on minimizing the Bregman divergence with respect to the Bernoulli distribution, which is suitable for datasets where each entry represents a binary outcome (e.g., 0 or 1).

# Arguments
- `indim::Integer`: The dimensionality of the input data (number of features). This represents the original high-dimensional space of the binary data.

- `outdim::Integer`: The dimensionality of the output (reduced space). This is the target lower-dimensional representation the data will be compressed into.

# Keyword Arguments
- `μ::Real = 0.5`: The regularization parameter representing the expected value of the Bernoulli distribution. Must be in the range `(0, 1)` as it corresponds to a probability. Defaults to `0.5`.

- `ϵ::Real = eps()`: A small positive value added for numerical stability. This helps in preventing divisions by zero or logarithms of zero during the optimization process. Defaults to `eps()`, which is the smallest representable positive number such that `1.0 + eps()` is distinguishable from `1.0`.

# Returns
- `epca::EPCA`: An instance of the EPCA model initialized for Bernoulli-distributed data. This model can be fitted to binary data using functions like `fit!`, and can then be used for data compression or reconstruction.

# Examples
```julia
# Import the module
using ExpFamilyPCA

# Define the EPCA model parameters
indim = 100  # Input dimension (number of features)
outdim = 10  # Reduced output dimension (number of principal components)

# Create a Bernoulli EPCA model instance
bernoulli_epca_model = BernoulliEPCA(indim, outdim; μ=0.5, ϵ=1e-6)

# Generate some random binary data
X = rand(Bool, 1000, indim)  # 1000 observations, each with 100 features (binary)

# Fit the model to the binary data
A = fit!(bernoulli_epca_model, X; maxiter=200, verbose=true)

# The resulting matrix A is the lower-dimensional representation of X
```

# Notes
- The link function ``g(θ) = exp(θ) / (1 + exp(θ))`` is the sigmoid function, which is appropriate for modeling Bernoulli-distributed data.
- The function ``Bg(x, θ) = B_F(x, g(θ))`` computes the Bregman divergence for the Bernoulli distribution.
- The choice of `μ`` must be within (0, 1) to ensure it is in the valid range for a probability under the Bernoulli distribution.
"""
function BernoulliEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 0.5,
    ϵ = eps()
)
    xs(x) = 2x - 1
    Bg(x, θ) = log1p(exp(-xs(x) * θ))
    g(θ) = exp(θ) / (1 + exp(θ))
    @assert 0 < μ < 1 "For BernoulliEPCA, μ must between (0, 1) to be in the range of g(θ) = sigmoid(θ)."
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        μ = μ,
        ϵ = ϵ
    )
    return epca
end