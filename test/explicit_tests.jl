function test_explicit(model::Function, X, rtol)
    @testset "$model" begin
        n, d = size(X)
        epca = model()
        X̃1 = fit!(epca, X; maxoutdim=d)
        X̃2 = compress(epca, X; maxoutdim=d)
        # @show X̃1
        # @show X̃2
        @test isapprox(X̃1, X̃2, rtol=rtol)
        X̂1 = decompress(epca, X̃1)
        X̂2 = decompress(epca, X̃2)
        # @show X̂1 
        # @show X̂2
        @test isapprox(X̂1, X̂2, rtol=rtol)
    end
end

@testset "Explicit Models" begin
    n = 10
    d = 5
    test_explicit(NormalEPCA, rand(n, d) * 100, 1)
    test_explicit(PoissonEPCA, rand(0:100, n, d), 1)
    test_explicit(BernoulliEPCA, rand(0:1, n, d), 1)  # NOTE: this is an admittedly generous tolerance
end