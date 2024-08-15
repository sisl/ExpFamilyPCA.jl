module ExpFamilyPCA

using Infiltrator

using Distances
using Symbolics
using Optim

export
    EPCA,
    fit!,
    compress,
    decompress
include("epca.jl")

include("utils.jl")
include("constructors/epca1.jl")
include("constructors/epca2.jl")
include("constructors/epca3.jl")


export
    NormalEPCA,
    BernoulliEPCA,
    PoissonEPCA,
    ItakuraSaitoEPCA,
    MahalanobisEPCA
include("family/normal.jl")
include("family/bernoulli.jl")
include("family/poisson.jl")
include("family/itakura_saito.jl")
include("family/mahalanobis.jl")

end # module ExpFamilyPCA
