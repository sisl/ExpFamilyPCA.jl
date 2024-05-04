function test_explicit(model::Function, X, rtol)
    @testset "$model" begin
        _, d = size(X)
        epca = model()
        Y1 = fit!(epca, X; maxoutdim=d)
        Y2 = compress(epca, X)
        @test isapprox(Y1, Y2, rtol=rtol)
        Z1 = decompress(epca, Y1)
        Z2 = decompress(epca, Y2)
        @test isapprox(Z1, Z2, rtol=rtol)
    end
end

@testset "Explicit Models" begin
    n = 10
    d = 5
    test_explicit(NormalEPCA, rand(n, d) * 100, 1)
    test_explicit(PoissonEPCA, rand(0:100, n, d), 1)
    test_explicit(BernoulliEPCA, rand(0:1, n, d), 1)
end