function NormalEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps()
)
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    Bregman(p, q) = Distances.sqeuclidean(p, q) / 2  # TODO: ask Mykel why /2 is necessary for stability
    g = identity
    epca = EPCA(
        indim,
        outdim,
        Bregman,
        g,
        Val((:Bregman, :g));
        μ=μ,
        ϵ=ϵ
    )
    return epca
end

# alias
function GaussianEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps()
)
    epca = NormalEPCA(
        indim,
        outdim;
        μ=μ,
        ϵ=ϵ
    )
    return epca
end