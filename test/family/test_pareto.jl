@testset "Pareto" begin
    m = 2
    n = 2
    indim = 3
    outdim = indim
    X = (rand(indim, outdim) * 2) .+ m

    M1 = ParetoEPCA(indim, outdim, m)
    A1 = fit!(M1, X)

    smoke_test(
        M1,
        A1,
        X,
        atol=1
    )
end
