"""
    BinomialEPCA(indim::Integer, outdim::Integer, n::Integer; options = Options(μ = 0.5))

Binomial EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `n::Integer`: A known parameter of the distribution representing the number of trials. Must be nonnegative.
- `options`: Optional parameters.

# Returns
- `epca`: A model instance of type `EPCA`.
"""
function BinomialEPCA(
    indim::Integer, 
    outdim::Integer, 
    n::Integer;
    options = Options(μ = 0.5)
)
    @assert n >= 0 "Number of trials n must be nonnegative."
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