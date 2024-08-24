@testset "Gamma" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(n, indim) * 100

    test_epca(
        "Gamma",
        GammaEPCA(indim, outdim),
        X,
        atol=0.5
    )

    F(x) = -x * log(x)
    g(θ) = -1 / θ

    ϵ = eps()
    G(θ) = -log(-θ)
    g(θ) = -1 / (θ + ϵ)
    F(x) = -1 - log(x + ϵ)
    f(x) = -1 / (x + ϵ)
    Bregman(p, q) = G(p) - G(p) - (p - q) * g(p)
    μ = -1

    @testset "EPCA1" begin
        test_equivalence(
            "F, g",
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                g,
                Val((:F, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )


        test_equivalence(
            "F, f",
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                f,
                Val((:F, :f));
                μ=μ,
                ϵ=ϵ,
                low=eps()
            ),
            X
        )

        test_equivalence(
            "F",
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                Val((:F));
                μ=μ,
                ϵ=ϵ,
                low=eps()
            ),
            X
        )

        test_equivalence(
            "F, G",
            GammaEPCA(indim, outdim),
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
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                G,
                g,
                Val((:G, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "G",
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                G,
                Val((:G));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )
    end

    @testset "EPCA3" begin
        test_equivalence(
            "Bregman, g",
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman,
                g,
                Val((:Bregman, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "Bregman, G",
            GammaEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman,
                G,
                Val((:Bregman, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )
    end
end
