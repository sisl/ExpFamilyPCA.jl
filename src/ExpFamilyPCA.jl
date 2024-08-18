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
    PoissonEPCA
include("family/poisson.jl")


# export
#     BernoulliEPCA,
#     BinomialEPCA,
#     ExponentialEPCA,
#     GammaEPCA,
#     ItakuraSaitoEPCA,
#     GeometricEPCA,
#     HyperbolicSecantEPCA,
#     InverseGaussianEPCA,
#     NegativeBinomialEPCA,
#     NormalEPCA,
#     GaussianEPCA,
#     PoissonEPCA
# include("family/normal.jl")
# include("family/bernoulli.jl")
# include("family/poisson.jl")
# include("family/gamma.jl")

# export
#     MahalanobisEPCA
# include("bregman/mahalanobis.jl")

end # module ExpFamilyPCA
