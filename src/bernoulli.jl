"""
Best with binary data. 
"""
function BernoulliPCA(l::Int; μ0::Real=0.5, kwargs...)
    epca = EPCA(l, μ0; kwargs...)
    # TODO: eventually replace this w/ symbolic diff
    ϵ = 10e-20
    @. begin
        G(θ) = log(1 + exp(θ))
        g(θ) = exp(θ) / (1 + exp(θ))
        F(x) = x * log(x) + (1 - x) * log(1 - x)
        f(x) = log(x / (1 - x))
        # TODO: look into when this value is negative
        Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + (1 - p) * log((1 - p + ϵ) / (1 - q + ϵ))  # with additive smoothing
    end
    epca.G = G
    epca.g = g
    epca.F = F
    epca.f = f
    epca.Bregman = Bregman
    return epca
end