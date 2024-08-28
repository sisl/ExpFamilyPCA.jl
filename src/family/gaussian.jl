function NormalEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
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
        μ = μ,
        ϵ = ϵ
    )
    return epca
end

# Alias
function GaussianEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
)
    epca = NormalEPCA(
        indim,
        outdim;
        μ = μ,
        ϵ = ϵ
    )
    return epca
end