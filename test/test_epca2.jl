@testset ":G Models" begin
    n = 2
    d = 5
    l = d

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

    @testset "No Metaprogramming" begin
        simple_smoke_check(
            "Normal", 
            EPCA(
                d, 
                l, 
                x -> x^2 / 2, 
                Val(:G), 
                metaprogramming = false
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
                Val(:G), 
                metaprogramming = false
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
                Val(:G), 
                metaprogramming = false
            ), 
            rand(0:1, n, d), 
            0.5
        )
    end
end
