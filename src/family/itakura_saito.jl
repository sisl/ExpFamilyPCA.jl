function ItakuraSaitoEPCA(
    indim::Integer,
    outdim::Integer;
    ϵ=eps()
)
    # χ = ℝ++

    @. begin
        F(x) = -x * log(x)
        g(θ) = -1 / θ
    end
    μ = g(1)
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
