"""
    NegativeBinomialEPCA(indim::Integer, outdim::Integer, r::Integer; options::Options = Options(A_init_value = -1, A_upper = -eps(), V_lower = eps()))

Negative binomial EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `r::Integer`: A known parameter of the negative binomial distribution representing the number of successes (positive).
- `options::Options`: Optional configuration parameters for the EPCA model. 
    - `A_init_value`: Initial fill value for matrix `A` (default: `-1`).
    - `A_upper`: The upper bound for the matrix `A`, default is `-eps()`.
    - `V_lower`: The lower bound for the matrix `V`, default is `eps()`.

!!! tip
    Try using `options = NegativeDomain()` if you encounter domain errors when calling `fit!` or `compress`.

# Returns
- `epca`: An instance of an `EPCA` subtype.
"""
function NegativeBinomialEPCA(
    indim::Integer, 
    outdim::Integer, 
    r::Integer;
    options::Options = Options(
        A_init_value = -1.0,
        A_upper = -eps(),
        V_lower = eps()
    )
)
    @assert r > 0 "Number of successes r must be positive."
    Bg(x, θ) = -r * log1mexp(θ) - x * θ
    g(θ) = xexpy(-r, θ) / expm1(θ)
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
