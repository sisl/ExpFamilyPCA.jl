# TODO: double check math

# NOTE: reference https://jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf (page 5, table 1)

"""
A is PSD [technically invertibility requires strict PD]
Math:
    F = x^T * A * x
    f = 2Ax (since A is symmetric)
    g = 1/2 * A^-1 x (A is invertible)
"""

function MahalanobisEPCA(
    indim::Integer,
    outdim::Integer,
    Q::AbstractMatrix;
    ϵ=eps()
)
    # NOTE: equivalent to generic PCA
    # assume χ = ℝ
    @. begin
        Bregman(p, q) = Distances.mahalanobis(p, q, Q)
    end
    g(θ) = 1/2  * inv(Q) * θ  # TODO: double check this math
    μ = g(1)
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