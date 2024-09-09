"""
    GammaEPCA(indim::Integer, outdim::Integer; options = Options(μ = 0.5))

Gamma EPCA.

TODO: finish optional parameters

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options`: Optional parameters.

# Returns
- `epca`: A model instance of type `EPCA`.
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
        A_init_value = -1,
        A_upper = -eps(),
        V_lower = eps()
    )
)
    epca = GammaEPCA(indim, outdim; options = options)
    return epca
end