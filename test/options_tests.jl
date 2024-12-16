@testset "Misc Options Tests" begin
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
    μ = 1

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
        X;
        custom_options = Options(A_use_sobol=true, V_use_sobol=true)
    )

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
        X;
        custom_options = Options(metaprogramming=false)
    )

    M1 = GaussianEPCA(indim, outdim; options=Options(A_lower=-1000., A_upper=1000.))
    A1 = fit!(M1, X)
    smoke_test(
        M1,
        A1,
        X,
        atol=0.5
    )
end

@testset "Positive Domain" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(-100:100, n, indim)

    M1 = GaussianEPCA(indim, outdim; options=PositiveDomain())
    A1 = fit!(M1, X)
    Θ = A1 * M1.V
    @test all(Θ .> 0)
end

@testset "Negative Domain" begin
    @testset "Negative Binomial" begin
        r = 2
        n = 2
        indim = 3
        outdim = 3
        X = rand(1:10, indim, outdim)
    
        M1 = NegativeBinomialEPCA(indim, outdim, r; options=NegativeDomain())
        A1 = fit!(M1, X)
        Θ = A1 * M1.V
        @test all(Θ .< 0)
    end

    @testset "Gamma" begin
        n = 2
        indim = 5
        outdim = 5
        X = rand(n, indim) * 100
    
        M1 = GammaEPCA(indim, outdim; options=NegativeDomain())
        A1 = fit!(M1, X)
        Θ = A1 * M1.V
        @test all(Θ .< 0)
    end

    @testset "Weibull" begin
        n = 2
        indim = 3
        outdim = 3
        X = rand(indim, outdim) * 2.5
    
        M1 = WeibullEPCA(indim, outdim; options=NegativeDomain())
        A1 = fit!(M1, X)
        Θ = A1 * M1.V
        @test all(Θ .< 0)
    end
end