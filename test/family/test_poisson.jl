@testset "Poisson" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:100, n, indim)

    test_epca(
        "Poisson",
        PoissonEPCA(indim, outdim),
        X,
        atol=0.5
    )

    ϵ = eps()
    G(θ) = exp(θ)
    g(θ) = exp(θ)
    F(x) = x * log(x + ϵ) - x
    f(x) = log(x + ϵ)
    Bregman1 = Distances.gkl_divergence
    Bregman2(p, q) = p * log(p / (q + ϵ) + ϵ) + q - p
    μ = g(0)

    @testset "EPCA1" begin
        test_equivalence(
            "F, g",
            PoissonEPCA(indim, outdim),
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
            PoissonEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                f,
                Val((:F, :f));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "F",
            PoissonEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                F,
                Val((:F));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "F, G",
            PoissonEPCA(indim, outdim),
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
            PoissonEPCA(indim, outdim),
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
            PoissonEPCA(indim, outdim),
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
            "Bregman1, g",
            PoissonEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman1,
                g,
                Val((:Bregman, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "Bregman2, g",
            PoissonEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman2,
                g,
                Val((:Bregman, :g));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "Bregman1, G",
            PoissonEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman1,
                G,
                Val((:Bregman, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )

        test_equivalence(
            "Bregman2, G",
            PoissonEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman2,
                G,
                Val((:Bregman, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X
        )
    end
end
