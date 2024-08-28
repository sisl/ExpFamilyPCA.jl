"""
NOTE: The product A*V must contain only negtive entries.
"""
function GammaEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps()
)
    # χ = ℝ++
    Bg(x, θ) = -x * θ - log(-x * θ) - 1
    g(θ) = -1 / θ
    @assert μ ≠ 0 "For GammaEPCA, μ must be nonzero to be in the range of g(θ) = -1/θ."
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        μ = μ,
        ϵ = ϵ,
        A_init_value = -1,
        A_upper = -eps(),
        V_lower = eps(),
    )
    return epca
end

# Alias
function ItakuraSaitoEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
)
    epca = GammaEPCA(
        indim,
        outdim;
        μ = μ,
        ϵ = ϵ
    )
    return epca
end