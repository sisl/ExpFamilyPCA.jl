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
        @test isapprox(Z1, Z2, atol=atol)
        @test isapprox(Z1, X, atol=atol)
        @test isapprox(Z2, X, atol=atol)
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
    μ::Real,
    ϵ::Real, 
    X::AbstractMatrix;
    A_init_value = nothing,
    A_lower = nothing,
    A_upper = nothing,
    V_init = nothing,
    V_lower = nothing,
    V_upper = nothing,
    # binary search parameters
    low = -1e10,
    high = 1e10,
    tol = 1e-10,
    maxiter = 1e6,
    epca1 = true,
    epca2 = true,
    epca3 = true,
    epca4 = true,
    epca5 = true
)
    M1 = epca_constructor(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=0.5
    )

    if epca1
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
                    μ = μ,
                    ϵ = ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper,
                    low = low, 
                    high = high, 
                    tol = tol, 
                    maxiter = maxiter
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper,
                    low = low, 
                    high = high, 
                    tol = tol, 
                    maxiter = maxiter
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
                ),
                X
            )
        end
    end

    if epca2
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
                ),
                X
            )
        end
    end

    if epca3
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
                ),
                X
            )
        end
    end

    if epca4
        @testset "EPCA4" begin
            test_equivalence(
                "f, G, g",
                M1,
                A1,
                EPCA(
                    indim,
                    outdim,
                    f,
                    G,
                    g,
                    Val((:f, :G, :g));
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
                ),
                X
            )

            test_equivalence(
                "F, g",
                M1,
                A1,
                EPCA(
                    indim,
                    outdim,
                    f,
                    G,
                    Val((:f, :G));
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
                ),
                X
            )
        end
    end

    if epca5
        @testset "EPCA5" begin
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
                    μ=μ,
                    ϵ=ϵ,
                    V_init = V_init,
                    A_init_value = A_init_value,
                    A_lower = A_lower,
                    A_upper = A_upper,
                    V_lower = V_lower,
                    V_upper = V_upper
                ),
                X;
                rtol=1e-7
            )
        end
    end
end
