"""
    BinomialEPCA(indim::Integer, outdim::Integer, n::Integer; options::Options = Options(μ = 0.5))

Binomial EPCA.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `n::Integer`: A known parameter representing the number of trials (nonnegative).
- `options::Options`: Optional parameters (default: `μ = 0.5`).

# Returns
- `epca`: An `EPCA` subtype for the binomial distribution.
"""
function BinomialEPCA(
    indim::Integer, 
    outdim::Integer, 
    n::Integer;
    options::Options = Options(
        μ = 0.5
    )
)
    @assert n >= 0 "Number of trials n must be nonnegative."
    @assert 0 < options.μ < n "μ must be in the range of the scaled logistic (0, n)."
    Bg(x, θ) = n * log1pexp(θ) - x * θ
    g(θ) = n * logistic(θ)
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = options
    )
    return epca
end