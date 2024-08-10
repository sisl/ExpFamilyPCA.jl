function test_explicit(name, epca::EPCA, X, rtol)
    @testset "$name" begin
        Y1 = fit!(epca, X)
        Y2 = compress(epca, X)
        @test isapprox(Y1, Y2, rtol=rtol)
        Z1 = decompress(epca, Y1)
        Z2 = decompress(epca, Y2)
        @test isapprox(Z1, Z2, rtol=rtol)
        @test isapprox(Z1, X, rtol=rtol)
        @test isapprox(Z2, X, rtol=rtol)
    end
end

@testset "Explicit Models" begin
    n = 2
    d = 5
    l = d
    test_explicit(
        "Normal", 
        NormalEPCA(d, l), 
        rand(n, d) * 100, 
        1
    )
    test_explicit(
        "Itakura-Saito", 
        ItakuraSaitoEPCA(d, l), 
        rand(n, d) * 100, 
        1
    )
    test_explicit(
        "Poisson",
        PoissonEPCA(d, l),
        rand(0:100, n, d),
        1
    )
    test_explicit(
        "Bernoulli",
        BernoulliEPCA(d, l),
        rand(0:1, n, d),
        0.5
    )

    # TODO: add no metapgoramming test
end