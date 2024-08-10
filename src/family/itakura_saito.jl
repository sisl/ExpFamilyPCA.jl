function ItakuraSaitoEPCA(
    indim::Integer,
    outdim::Integer;
    ϵ=eps()
)
    @. begin
        F(x) = x * log(x + ϵ)
        f(x) = 1 + log(x + ϵ)
        g(θ) = exp(θ - 1)
    end
    μ = g(1)
    epca = EPCA(
        indim,
        outdim,
        F,
        f,
        g,
        Val((:F, :f, :g));
        μ=μ,
        ϵ=ϵ
    )
    return epca
end
