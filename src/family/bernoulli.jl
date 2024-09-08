"""
    BernoulliEPCA(indim::Integer, outdim::Integer; options = Options(μ = 0.5))

Bernoulli EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options`: Optional parameters.

# Returns
- `epca`: A model instance of type `EPCA`.
"""
function BernoulliEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options(
        μ = 0.5
    )
)
    @assert 0 < options.μ < 1 "μ must be in the range of the logistic (0, 1)."
    xs(x) = 2x - 1
    Bg(x, θ) = log1pexp(-(2x - 1) * θ)
    g = logistic
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