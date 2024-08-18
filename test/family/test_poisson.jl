@testset "Poisson" begin
    n = 2
    d = 4
    l = d

    # TODO: add other constructors
    @testset "Metaprogramming" begin
        simple_smoke_check(
            "Normal", 
            EPCA(
                d, 
                l, 
                x -> x^2 / 2, 
                Val(:G)
            ), 
            rand(n, d) * 100, 
            1
        )
        simple_smoke_check(
            "Poisson", 
            EPCA(
                d, 
                l, 
                x -> exp(x), 
                Val(:G)
            ), 
            rand(0:100, n, d), 
            1
        )
        simple_smoke_check(
            "Bernoulli", 
            EPCA(
                d, 
                l, 
                x -> exp(x) / (1 + exp(x)), 
                Val(:G)
            ), 
            rand(0:1, n, d), 
            0.5
        )
    end
end
