@testset "Bernoulli" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:1, n, indim)

    M1 = BernoulliEPCA(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=0.01
    )
end
