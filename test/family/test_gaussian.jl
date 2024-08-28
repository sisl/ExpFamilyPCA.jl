@testset "Gaussian" begin
    n = 2
    indim = 5
    outdim = 5
    X = (rand(n, indim) .- 0.5) .* 100

    ϵ = eps()
    G(θ) = θ^2 / 2
    g = identity
    F = G
    f = identity
    B(p, q) = Distances.sqeuclidean(p, q) / 2
    Bg = B
    μ = g(1)

    run_EPCA_tests(
        GaussianEPCA,
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
