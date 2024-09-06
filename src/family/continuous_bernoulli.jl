"""
hi
"""
function ContinuousBernoulliEPCA(indim::Integer, outdim::Integer)
    Bg(x, θ) = log(expm1(θ) / θ) - x * θ
    g(θ) = (θ - 1) / θ + 1 / expm1(θ)  # TODO: maybe find simpler version of this
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
    )
    return epca
end