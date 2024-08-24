function GammaEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps(),
)
    # χ = ℝ++
    @. begin
        F(x) = -1 - log(x + ϵ)
        g(θ) = -1 / (θ + ϵ)
    end
    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g));
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