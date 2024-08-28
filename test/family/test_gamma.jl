@testset "Gamma" begin
    n = 2
    indim = 5
    outdim = 5
    X = rand(n, indim) * 10

    M1 = GammaEPCA(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=0.01
    )
end
