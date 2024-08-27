@testset "Bernoulli" begin
    n = 2
    indim = 3
    outdim = 5
    X = rand(0:1, n, indim)

    ϵ = eps()
    G(θ) = log1p(exp(θ))
    g(θ) = exp(θ) / (1 + exp(θ))
    function F(x)
        if x == 0
            return 0
        elseif x == 1
            return 
        x * log(x + ϵ) + (1 - x) * log(1 - x + ϵ)
    f(x) = log(x) - log(1 - x + ϵ)
    Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p
    μ = 0.5
    V_init = nothing
    A_init_value = -1
    A_lower = -Inf
    A_upper = -ϵ
    V_lower = ϵ
    V_upper = Inf

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
