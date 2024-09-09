"""
    GaussianEPCA(indim::Integer, outdim::Integer; options = Options(μ = 0))

Gaussian EPCA.

TODO add optional parameters

# Arguments
- `indim::Integer`: The dimension of the input space.
- `outdim::Integer`: The dimension of the latent (output) space.
- `options`: Optional parameters.

# Returns
- `epca`: A model instance of type `EPCA`.
"""
function NormalEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options(
        V_init_value = 0
    )
)
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    B(p, q) = Distances.sqeuclidean(p, q) / 2
    g = identity
    epca = EPCA(
        indim,
        outdim,
        B,
        g,
        Val((:B, :g));
        options = options
    )
    return epca
end

"""
Alias for [`NormalEPCA`](@ref).
"""
function GaussianEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options(
        V_init_value = 0
    )
)
    epca = NormalEPCA(indim, outdim; options = options)
    return epca
end