@testset "Bernoulli" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:1, n, indim)

    ϵ = eps()
    G(θ) = log1p(exp(θ))
    g(θ) = exp(θ) / (1 + exp(θ))
    F(x) = x * log(x + ϵ) + (1 - x) * log(1 - x + ϵ)
    f(x) = log((x + ϵ) / (1 - x + ϵ))
    Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p
    μ = 0.5

    run_EPCA_tests(
        BernoulliEPCA,
        indim,
        outdim,
        F,
        G,
        f,
        g,
        Bregman,
        μ,
        ϵ,
        X
    )
end
