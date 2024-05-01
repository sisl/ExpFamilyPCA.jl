using ExpFamilyPCA
using Test

using Random
Random.seed!(1)


@testset "ExpFamilyPCA.jl" begin
    include("explicit_tests.jl")
end
