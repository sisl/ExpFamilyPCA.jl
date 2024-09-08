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
function GammaEPCA(indim::Integer, outdim::Integer)
    # χ = ℝ++
    Bg(x, θ) = -log(-θ) - x * θ
    g(θ) = -1 / θ
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = Options(
            A_init_value = -1,
            A_upper = -eps(),
            V_lower = eps()
        )
    )
    return epca
end

"""
Alias for [`GammaEPCA`](@ref).
"""
function ItakuraSaitoEPCA(indim::Integer, outdim::Integer)
    epca = GammaEPCA(indim, outdim)
    return epca
end