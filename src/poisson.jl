function PoissonPCA(l::Int; μ0::Real=0, kwargs...)
    epca = EPCA(l, μ0; kwargs...)
    # TODO: eventually replace this w/ symbolic diff
    # ϵ = 10e-20
    ϵ = eps()
    @. begin
        G(θ) = exp(θ)
        g(θ) = exp(θ)
        F(x) = x * log(x) - x
        f(x) = log(x)
        Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p  # with additive smoothing
    end
    epca.G = G
    epca.g = g
    epca.F = F
    epca.f = f
    epca.Bregman = Bregman
    return epca
end


# TODO: include a normalized Poisson w/ link function in footnote 5 of long paper
