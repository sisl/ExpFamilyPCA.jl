function simple_smoke_check(
    name::String, 
    epca::EPCA, 
    X, 
    rtol
)
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