"""
    PoissonEPCA(indim::Integer, outdim::Integer; options::Options = Options())

An EPCA model with Poisson loss.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `options::Options`: Optional parameters.

# Returns
- `epca`: An `EPCA` subtype for the Poisson distribution.
"""
function PoissonEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options()
)
    @assert options.μ > 0 "μ must be in the range of the exponential (0, ∞)."

    # assumes χ = ℕ
    Bg(x, θ) = exp(θ) - x * θ
    g = exp
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

