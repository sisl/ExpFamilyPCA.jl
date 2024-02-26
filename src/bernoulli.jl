# TODO: add caveat that this works best w/ data in [0, 1] and might fail otherwise
function BernoulliPCA(l::Integer, d::Integer; μ0::Real=0.5)
    ϵ = eps()
    @. begin
        g(θ) = exp(θ) / (1 + exp(θ))
        Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + (1 - p) * log((1 - p + ϵ) / (1 - q + ϵ))  # with additive smoothing
    end
    epca = EPCA(l, d, g, Bregman, μ0)
    return epca
end