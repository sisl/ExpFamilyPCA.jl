@testset "Bernoulli" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:1, n, indim)

    test_epca(
        "Bernoulli",
        BernoulliEPCA(indim, outdim),
        X,
        atol=0.5
    )

    ϵ = eps()
    G(θ) = log1p(exp(θ))
    g(θ) = exp(θ) / (1 + exp(θ))
    F(x) = x * log(x + ϵ) + (1 - x) * log(1 - x + ϵ)
    f(x) = log((x + ϵ) / (1 - x + ϵ))
    Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p
    μ = 0.5

    @testset "EPCA1" begin
        test_equivalence(
            "F, g",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                g,
                Val((:F, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X;
            rtol=1e-3
        )


        test_equivalence(
            "F, f",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                f,
                Val((:F, :f));
                μ=μ,
                ϵ=ϵ,
                low=0,
                high=1
            ),
            X;
            rtol=1e-3
        )

        test_equivalence(
            "F",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                Val((:F));
                μ=μ,
                ϵ=ϵ,
                low=0,
                high=1
            ),
            X
        )

        test_equivalence(
            "F, G",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                G,
                Val((:F, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )
    end

    @testset "EPCA2" begin
        test_equivalence(
            "G, g",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                G,
                g,
                Val((:G, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X;
            rtol=1e-3
        )

        test_equivalence(
            "G",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                G,
                Val((:G));
                μ=μ,
                ϵ=ϵ
            ),
            X;
            rtol=1e-3
        )
    end

    @testset "EPCA3" begin
        test_equivalence(
            "Bregman, g",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman,
                g,
                Val((:Bregman, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X;
            rtol=1e-3
        )

        test_equivalence(
            "Bregman, G",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman,
                G,
                Val((:Bregman, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X;
            rtol=1e-3
        )
    end

    @testset "EPCA4" begin
        test_equivalence(
            "f, G, g",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                f,
                G,
                g,
                Val((:f, :G, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "F, g",
            BernoulliEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                f,
                G,
                Val((:f, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )
    end
end
