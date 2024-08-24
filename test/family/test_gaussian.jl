@testset "Gaussian" begin
    n = 2
    indim = 5
    outdim = 5
    X = (rand(n, indim) .- 0.5) .* 100

    test_epca(
        "Gaussian",
        GaussianEPCA(indim, outdim),
        X,
        atol=0.5
    )

    ϵ = eps()
    G(θ) = θ^2 / 2
    g(θ) = identity(θ)
    F(x) = x^2 / 2
    f(x) = identity(x)
    Bregman1(p, q) = Distances.sqeuclidean(p, q) / 2
    Bregman2(p, q) = (p - q)^2 / 2
    μ = g(1)

    @testset "EPCA1" begin
        test_equivalence(
            "F, g",
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
            EPCA(
                indim,
                outdim,
                Bregman1,
                G,
                Val((:Bregman, :G));
                μ=μ,
                ϵ=ϵ
            ),
            X;
        )

        test_equivalence(
            "Bregman2, G",
            GaussianEPCA(indim, outdim),
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

    @testset "EPCA4" begin
        test_equivalence(
            "f, G, g",
            GaussianEPCA(indim, outdim),
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
            GaussianEPCA(indim, outdim),
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
