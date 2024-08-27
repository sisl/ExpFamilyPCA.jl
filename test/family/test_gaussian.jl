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
    Bregman(p, q) = Distances.sqeuclidean(p, q) / 2
    μ = g(1)

    run_EPCA_tests(
        GaussianEPCA,
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
