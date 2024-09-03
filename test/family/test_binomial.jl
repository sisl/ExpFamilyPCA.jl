@testset "Binomial" begin
    trials = 10
    n = 2
    indim = 5
    outdim = 5
    X = rand(0:trials, n, indim)

    M1 = BinomialEPCA(indim, outdim, trials)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=0.01
    )
end
