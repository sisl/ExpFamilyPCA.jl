# function GammaEPCA(
#     indim::Integer,
#     outdim::Integer;
#     μ=1,
#     ϵ=eps(),
# )
#     # χ = ℝ++
#     F(x) = -1 - log(x + ϵ)
#     g(θ) = -1 / (θ + ϵ)
#     epca = EPCA(
#         indim,
#         outdim,
#         F,
#         g,
#         Val((:F, :g));
#         μ=μ,
#         ϵ=ϵ
#     )
#     return epca
# end

function GammaEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps(),
)
    # χ = ℝ++
    Bregman(p, q) = p / q - log((p + ϵ) / (q + ϵ)) - 1
    g(θ) = -1 / (θ + ϵ)
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

# Alias
function ItakuraSaitoEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps()
)
    epca = GammaEPCA(
        indim,
        outdim;
        μ=μ,
        ϵ=ϵ
    )
    return epca
end