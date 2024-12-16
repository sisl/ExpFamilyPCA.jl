@testset "Gamma" begin
    n = 2
    indim = 3
    outdim = 3
    X = rand(n, indim) * 100

    M1 = GammaEPCA(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=1.5
    )
end

@testset "ItakuraSaito" begin
    n = 2
    indim = 3
    outdim = 3
    X = rand(n, indim) * 100

    M1 = ItakuraSaitoEPCA(indim, outdim)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=1.5
    )
end

