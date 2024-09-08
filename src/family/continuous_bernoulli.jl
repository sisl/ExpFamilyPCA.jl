"""
    ContinuousBernoulliEPCA(indim::Integer, outdim::Integer; options = Options())

Continuous Bernoulli EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options`: Optional parameters.

# Returns
- `epca`: A model instance of type `EPCA`.
"""
function ContinuousBernoulliEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options()
)
    Bg(x, θ) = log(expm1(θ) / θ) - x * θ
    g(θ) = (θ - 1) / θ + 1 / expm1(θ)  # TODO: maybe find simpler version of this
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