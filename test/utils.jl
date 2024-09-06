function is_constant_matrix(A::AbstractMatrix)
    flag = length(unique(A)) == 1
    return flag
end

function smoke_test(
    epca::EPCA,
    A1::AbstractMatrix,
    X::AbstractMatrix;
    atol=0
)
    @testset "Smoke Test" begin
        A2 = @test_nowarn compress(epca, X)
        Z1 = @test_nowarn decompress(epca, A1)
        Z2 = @test_nowarn decompress(epca, A2)
        @test all(isapprox.(Z1, Z2, atol=atol))
        @test all(isapprox.(Z1, X, atol=atol))
        @test all(isapprox.(Z2, X, atol=atol))
        @test !is_constant_matrix(epca.V)
        @test !is_constant_matrix(epca.V)
        @test !is_constant_matrix(A1)
        @test !is_constant_matrix(A2)
        @test !is_constant_matrix(Z1)
        @test !is_constant_matrix(Z2)
    end
end

function test_equivalence(
    name::String, 
    M1::EPCA, 
    A1::AbstractMatrix,
    M2::EPCA, 
    X::AbstractMatrix; 
    rtol=1e-12
)
    @testset "$name" begin
        A2 = @test_nowarn fit!(M2, X)
        @test isapprox(A1, A2; rtol=rtol)
        B1 = @test_nowarn compress(M1, X)
        B2 = @test_nowarn compress(M2, X)
        @test isapprox(B1, B2, rtol=rtol)
        Z1 = @test_nowarn decompress(M1, A1)
        Z2 = @test_nowarn decompress(M2, A2)
        @test isapprox(Z1, Z2, rtol=rtol)
    end
end

function run_EPCA_tests(
    epca_constructor::Function,
    indim::Integer, 
    outdim::Integer,
    Bg::Function,
    F::Function, 
    G::Function, 
    f::Function, 
    g::Function, 
    B::Union{Function, PreMetric}, 
    X::AbstractMatrix
)
    M1 = epca_constructor(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=0.5
    )

    options = M1.options

    @testset "EPCA1" begin
        test_equivalence(
            "F, g",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                F,
                g,
                Val((:F, :g));
                options = options
            ),
            X
        )


        test_equivalence(
            "F, f",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                F,
                f,
                Val((:F, :f));
                options = options
            ),
            X
        )

        test_equivalence(
            "F",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                F,
                Val((:F));
                options = options
            ),
            X
        )

        test_equivalence(
            "F, G",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                F,
                G,
                Val((:F, :G));
                options = options
            ),
            X
        )
    end

    @testset "EPCA2" begin
        test_equivalence(
            "G, g",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                G,
                g,
                Val((:G, :g));
                options = options
            ),
            X
        )

        test_equivalence(
            "G",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                G,
                Val((:G));
                options = options
            ),
            X
        )
    end

    @testset "EPCA3" begin
        test_equivalence(
            "B, g",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                B,
                g,
                Val((:B, :g));
                options = options
            ),
            X
        )

        test_equivalence(
            "B, G",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                B,
                G,
                Val((:B, :G));
                options = options
            ),
            X
        )
    end

    @testset "EPCA4" begin
        test_equivalence(
            "Bg, g",
            M1,
            A1,
            EPCA(
                indim,
                outdim,
                Bg,
                g,
                Val((:Bg, :g));
                options = options
            ),
            X;
            rtol=1e-7
        )
    end
end
