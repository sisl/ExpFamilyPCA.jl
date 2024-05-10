function test_equivalence(name, M1::EPCA, M2::EPCA, X; rtol=eps())
    @testset "$name" begin
        A1 = fit!(M1, X)
        A2 = fit!(M2, X)
        @test isapprox(A1, A2; rtol=rtol)
        B1 = compress(M1, X)
        B2 = compress(M2, X)
        @test isapprox(B1, B2, rtol=rtol)
        Z1 = decompress(M1, A1)
        Z2 = decompress(M2, A2)
        @test isapprox(Z1, Z2, rtol=rtol)
    end
end

@testset "Equivalence Sanity" begin
    n = 10
    d = 5
    test_equivalence("Normal", NormalEPCA(), EPCA(x->x^2/2), rand(n, d) * 100)
    test_equivalence("Poisson", PoissonEPCA(), EPCA(x->exp(x)), rand(0:100, n, d))
    # test_equivalence("Bernoulli", BernoulliEPCA(), EPCA(x->exp(x)/(1 + exp(x))), rand(0:1, n, d); rtol=0.75)
end