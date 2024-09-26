"""
    WeibullEPCA(indim::Integer, outdim::Integer; options::Options = Options(A_init_value = -1, A_upper = -eps(), V_lower = eps()))

Weibull EPCA.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `options::Options`: Optional parameters for model initialization:
    - `A_init_value`: Initial fill value for matrix `A` (default: `-1`).
    - `A_upper`: Upper bound for matrix `A` (default: `-eps()`).
    - `V_lower`: Lower bound for matrix `V` (default: `eps()`).

!!! tip
    Try using `options = NegativeDomain()` if you encounter domain errors when calling `fit!` or `compress`.

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