"""
    WeibullEPCA(indim::Integer, outdim::Integer; options::Options = Options(A_init_value = -1, A_upper = -eps(), V_lower = eps()))

Weibull EPCA.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `options::Options`: Optional parameters for model initialization. Default `NegativeDomain()`.

# Returns
- `epca`: An `EPCA` subtype for the Weibull distribution.
"""
function WeibullEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = NegativeDomain()
)
    Bg(x, θ) = -log(-θ) - x * θ
    g(θ) = -1 / θ
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