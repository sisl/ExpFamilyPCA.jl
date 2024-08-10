function PoissonEPCA(
    indim::Integer,
    outdim::Integer,
    ϵ=eps()
)
    # assumes χ = ℤ
    @. begin
        Bregman(p, q) = Distances.gkl_divergence(p, q)
        g(θ) = exp(θ)
    end
    μ = g(0)
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
