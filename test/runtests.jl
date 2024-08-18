using ExpFamilyPCA
using Test

using Distances
using Random
Random.seed!(1)  # used to generate random matrices for testing


include("utils.jl")


@testset "ExpFamilyPCA.jl" begin
    include("family/test_poisson.jl")
end
