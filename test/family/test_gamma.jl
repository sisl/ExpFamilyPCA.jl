@testset "Gamma" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(n, indim) * 10

    ϵ = eps()
    G(θ) = -log(-θ + ϵ)
    g(θ) = -1 / (θ + ϵ)
    F(x) = -1 - log(x + ϵ)
    f(x) = -1 / (x + ϵ)
    Bregman(p, q) =  p / q - log((p + ϵ) / (q + ϵ)) - 1  # Itakura-Saito Distance
    μ = 0
    V_init = nothing
    A_init_value = -1
    A_lower = -Inf
    A_upper = -ϵ
    V_lower = ϵ
    V_upper = Inf

    run_EPCA_tests(
        GammaEPCA,
        indim,
        outdim,
        F,
        G,
        f,
        g,
        Bregman,
        μ,
        ϵ,
        X;
        A_init_value = A_init_value,
        A_lower = A_lower,
        A_upper = A_upper,
        V_init = V_init,
        V_lower = V_lower,
        V_upper = V_upper
    )
end
