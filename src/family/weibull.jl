"""
    WeibullEPCA(indim::Integer, outdim::Integer; options::Options = Options(A_init_value = -1, A_upper = -eps(), V_lower = eps()))

Weibull EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options::Options`: Optional parameters for the model initialization:
    - `A_init_value`: Initial value for matrix `A`, defaults to `-1`.
    - `A_upper`: Upper bound for matrix `A`, defaults to `-eps()`.
    - `V_lower`: Lower bound for matrix `V`, defaults to `eps()`.

# Returns
- `epca`: A model instance of type `EPCA`.
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