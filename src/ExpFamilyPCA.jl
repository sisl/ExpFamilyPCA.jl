module ExpFamilyPCA

using Infiltrator

using FunctionWrappers: FunctionWrapper
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
include("constructors/epca4.jl")
include("constructors/epca5.jl")

export
    PoissonEPCA
include("family/poisson.jl")

export
    GaussianEPCA,
    NormalEPCA
include("family/gaussian.jl")

export
    BernoulliEPCA
include("family/bernoulli.jl")

export
    GammaEPCA,
    ItakuraSaitoEPCA
include("family/gamma.jl")


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

# TODO: export PoissonCompressor, etc.
export
    EPCACompressor
include("compressor.jl")

end # module ExpFamilyPCA
