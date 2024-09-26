"""
    GaussianEPCA(indim::Integer, outdim::Integer; options::Options = Options())

An EPCA model with Gaussian loss.

# Arguments
- `indim::Integer`: Dimension of the input space.
- `outdim::Integer`: Dimension of the latent (output) space.
- `options::Options`: Optional parameters.

# Returns
- `epca`: An EPCA subtype for the Gaussian distribution.
"""
function NormalEPCA(
    indim::Integer, 
    outdim::Integer;
    options::Options = Options()
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
    options::Options = Options()
)
    epca = NormalEPCA(indim, outdim; options = options)
    return epca
end