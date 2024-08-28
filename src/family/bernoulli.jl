
function BernoulliEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 0.5,
    ϵ = eps()
)
    xs(x) = 2x - 1
    Bg(x, θ) = log1p(exp(-xs(x) * θ))
    g(θ) = exp(θ) / (1 + exp(θ))
    
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        μ = μ,
        ϵ = ϵ
    )
    return epca
end