@testset "Poisson" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:100, n, indim)

    ϵ = eps()
    G = exp
    g = exp
    F(x) = xlogx(x) - x
    f(x) = log(x + ϵ)
    B = Distances.gkl_divergence
    Bg(x, θ) = exp(θ) - x * θ + xlogx(x) - x

    μ = 1

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
        X
    )
end
