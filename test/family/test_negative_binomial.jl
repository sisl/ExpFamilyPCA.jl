@testset "Negative Binomial" begin
    r = 2
    n = 2
    indim = 3
    outdim = 3
    X = rand(1:10, indim, outdim)

    M1 = NegativeBinomialEPCA(indim, outdim, r)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=3.5
    )
end
