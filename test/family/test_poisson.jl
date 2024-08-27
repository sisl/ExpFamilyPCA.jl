@testset "Poisson" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:100, n, indim)

    ϵ = eps()
    G(θ) = exp(θ)
    g(θ) = exp(θ)
    F(x) = x * log(x + ϵ) - x
    f(x) = log(x + ϵ)
    Bregman = Distances.gkl_divergence
    μ = g(0)

    run_EPCA_tests(
        PoissonEPCA,
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
