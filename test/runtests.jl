using ExpFamilyPCA
using Test

using Distances
using LogExpFunctions
using Random
Random.seed!(1)  # used to generate random matrices for testing


include("utils.jl")


@testset "ExpFamilyPCA.jl" begin
    include("family/test_poisson.jl")
    include("family/test_gaussian.jl")
    include("family/test_bernoulli.jl")
    include("family/test_gamma.jl")
    include("family/test_binomial.jl")
    include("family/test_continuous_bernoulli.jl")
    include("family/test_negative_binomial.jl")
    include("family/test_pareto.jl")
    include("family/test_weibull.jl")
end
