"""
    WeibullEPCA(indim::Integer, outdim::Integer; options::Options = Options(A_init_value = -1, A_upper = -eps(), V_lower = eps()))

An EPCA model with Weibull loss.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `options::Options`: Optional parameters for model initialization:
    - `A_init_value`: Initial fill value for matrix `A` (default: `-1`).
    - `A_upper`: Upper bound for matrix `A` (default: `-eps()`).
    - `V_lower`: Lower bound for matrix `V` (default: `eps()`).

# Returns
- `epca`: An `EPCA` subtype for the Weibull distribution.
"""
function WeibullEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options(
        A_init_value = -1,
        A_upper = -eps(),
        V_lower = eps()
    )
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