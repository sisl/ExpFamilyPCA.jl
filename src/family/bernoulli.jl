function BernoulliEPCA(
    indim::Integer,
    outdim::Integer;
    μ=0.5,
    ϵ=eps()
)
    # assumes χ = ℕ
    F(x) = x * log(x + ϵ) + (1 - x) * log1p(ϵ - x)
    g(θ) = exp(θ) / (1 + exp(θ))
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

