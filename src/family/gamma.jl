"""
    GammaEPCA(indim::Integer, outdim::Integer; options::Options = Options(A_init_value = -2, A_upper = -eps(), V_lower = eps()))

Gamma EPCA.

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options::Options`: Optional configuration parameters for the EPCA model. 
    - `A_init_value`: Initial fill value for matrix `A` (default: `-2`).
    - `A_upper`: The upper bound for the matrix `A`, default is `-eps()`.
    - `V_lower`: The lower bound for the matrix `V`, default is `eps()`.

# Returns
- `epca`: An instance of an `EPCA` subtype.
"""
function GammaEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options(
        A_init_value = -2,
        A_upper = -eps(),
        V_lower = eps()
    )
)
    # χ = ℝ++
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

"""
Alias for [`GammaEPCA`](@ref).
"""
function ItakuraSaitoEPCA(
    indim::Integer, 
    outdim::Integer; 
    options::Options = Options(
        A_init_value = -2,
        A_upper = -eps(),
        V_lower = eps()
    )
)
    epca = GammaEPCA(indim, outdim; options = options)
    return epca
end