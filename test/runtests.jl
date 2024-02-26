using ExpFamilyPCA
using Test

using Symbolics
using Random
Random.seed!(1)


function analytic_test(epca_constructor::Function, l::Integer, n::Integer, d::Integer)
    X = rand(n, d)
    epca = epca_constructor(l, d)
    fit!(epca, X, maxiter=50)
    X̃ = compress(X)
    X_recon = decompress(X̃)
    @test X ≈ X_recon
end


# TODO add numeric tests


@testset "ExpFamilyPCA.jl" begin
    include("poisson_tests.jl")
end
