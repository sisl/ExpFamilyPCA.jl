function test_implicit(name, epca::EPCA, X, rtol)
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

@testset "Implicit Models" begin
    n = 2
    d = 5
    l = d
    @testset "Metaprogramming" begin
        test_implicit("Normal", EPCA(d, l, x->x^2/2, Val(:G)), rand(n, d) * 100, 1)
        test_implicit("Poisson", EPCA(d, l, x->exp(x), Val(:G)), rand(0:100, n, d), 1)
        # test_implicit("Bernoulli", EPCA(d, l, x->exp(x)/(1 + exp(x)), Val(:G)), rand(0:1, n, d), 0.5)
    end

    @testset "No Metaprogramming" begin
        test_implicit("Normal", EPCA(d, l, x->x^2/2, Val(:G), metaprogramming=false), rand(n, d) * 100, 1)
        test_implicit("Poisson", EPCA(d, l, x->exp(x), Val(:G), metaprogramming=false), rand(0:100, n, d), 1)
        # test_implicit("Bernoulli", EPCA(d, l, x->exp(x)/(1 + exp(x)), Val(:G), metaprogramming=false), rand(0:1, n, d), 0.5)
    end
end