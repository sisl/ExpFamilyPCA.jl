module ExpFamilyPCA

# TODO: remove this
using Infiltrator

# export
#     EPCA,
#     fit!,
#     compress,
#     decompress
# include("faster_epca.jl")

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
