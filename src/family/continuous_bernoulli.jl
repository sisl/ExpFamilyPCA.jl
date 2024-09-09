"""
    ContinuousBernoulliEPCA(indim::Integer, outdim::Integer; options::Options = Options(μ = 0.5))

An EPCA model with continuous Bernoulli loss.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `options::Options`: Optional parameters (default: `μ = 0.25`).

# Returns
- `epca`: An `EPCA` subtype for the continuous Bernoulli distribution.
"""
function ContinuousBernoulliEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options(
        μ = 0.25
    )
)
    @assert 0 < options.μ < 1 && options.μ != 0.5
    Bg(x, θ) = log(expm1(θ) / θ) - x * θ
    g(θ) = (θ - 1) / θ + 1 / expm1(θ)
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