@testset "Weibull" begin
    n = 2
    indim = 3
    outdim = 3
    X = rand(indim, outdim) * 2.5

    M1 = WeibullEPCA(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=1
    )
end
