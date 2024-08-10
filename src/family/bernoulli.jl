function BernoulliEPCA(
    indim::Integer,
    outdim::Integer;
    ϵ=eps()
)
    # assumes χ = {0, 1}
    @. begin
        F(x) = x * log(x + ϵ) + (1 - x) * log(1 - x + ϵ)
        f(x) = log(x + ϵ) - log(1 - x + ϵ)
        g(θ) = exp(θ) / (1 + exp(θ))
    end
    μ = g(0)
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