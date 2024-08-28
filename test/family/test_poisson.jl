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
    B = Distances.gkl_divergence
    Bg(x, θ) = exp(θ) - x * θ + x * log(x) * x - x
    μ = g(0)

    run_EPCA_tests(
        PoissonEPCA,
        indim,
        outdim,
        Bg,
        F,
        G,
        f,
        g,
        B,
        μ,
        ϵ,
        X
    )
end
