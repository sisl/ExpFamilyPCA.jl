"""
    PoissonEPCA(indim::Integer, outdim::Integer; options = Options())

Poisson EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options`: Optional parameters.

# Returns
- `epca`: A model instance of type `EPCA`.
"""
function PoissonEPCA(
    indim::Integer, 
    outdim::Integer;
    options = Options()
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

