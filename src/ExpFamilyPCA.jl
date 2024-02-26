module ExpFamilyPCA

export
    EPCA,
    fit!,
    compress,
    decompress
include("epca.jl")

export
    PoissonPCA
include("poisson.jl")

export
    BernoulliPCA
include("bernoulli.jl")


end # module ExpFamilyPCA
