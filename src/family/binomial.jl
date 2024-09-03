"""
n = number of trials
"""
function BinomialEPCA(indim::Integer, outdim::Integer, n::Integer)
    @assert n > 0 "Number of trials n must be positive."
    Bg(x, θ) = n * log1pexp(θ) - x * θ
    g(θ) = n * logistic(θ)
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = Options(μ = 0.5)
    )
    return epca
end