"""
    ParetoEPCA(indim::Integer, outdim::Integer, m::Real; options::Options = Options(μ = 2, A_init_value = 2, A_lower = 1 / indim, V_init_value = -2, V_upper = -1))

Pareto EPCA.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `m::Real`: A known parameter of the Pareto distribution representing the minimum value in the support.
- `options::Options`: Optional parameters for model initialization:
    - `μ`: Default value `2`.
    - `A_init_value`: Initial value for matrix `A` (default: `2`).
    - `A_lower`: Lower bound for matrix `A` (default: `1 / indim`).
    - `V_init_value`: Initial value for matrix `V` (default: `-2`).
    - `V_upper`: Upper bound for matrix `V` (default: `-1`).

# Returns
- `epca`: An `EPCA` subtype for the Pareto distribution.
"""
function ParetoEPCA(
    indim::Integer, 
    outdim::Integer, 
    m::Real; 
    options::Options = Options(
        μ = 2,
        A_init_value = 2,
        A_lower = 1 / indim,
        V_init_value = -2,
        V_upper = -1,
    )
)
    @assert m > 0 "Minimum value m must be positive."
    @assert options.μ != log(m) "μ must be in the range of g."
    Bg(x, θ) = -log(-1 - θ) + xlogy(θ, m) - x * θ 
    g(θ) = log(m) - 1 / (θ + 1)
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = Options(
            μ = 2,
            A_init_value = 2,
            A_lower = 1 / indim,
            V_init_value = -2,
            V_upper = -1,
        )
    )
    return epca
end