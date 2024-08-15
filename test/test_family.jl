@testset "Named Models" begin
    n = 2
    d = 5
    l = d
    simple_smoke_check(
        "Normal", 
        NormalEPCA(d, l), 
        rand(n, d) * 100, 
        1
    )
    simple_smoke_check(
        "Itakura-Saito", 
        ItakuraSaitoEPCA(d, l), 
        rand(n, d) * 100, 
        1
    )
    simple_smoke_check(
        "Poisson",
        PoissonEPCA(d, l),
        rand(0:100, n, d),
        1
    )
    simple_smoke_check(
        "Bernoulli",
        BernoulliEPCA(d, l),
        rand(0:1, n, d),
        0.5
    )

    # TODO: add no metapgoramming test
end