module ExpFamilyPCA

using Infiltrator

using Distances
using FunctionWrappers: FunctionWrapper
using LogExpFunctions
using Optim
using Parameters
using Symbolics

using Statistics

export
    Options
include("options.jl")

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

export
    BinomialEPCA
include("family/binomial.jl")

export
    ContinuousBernoulliEPCA
include("family/continuous_bernoulli.jl")

export
    NegativeBinomialEPCA
include("family/negative_binomial.jl")

export
    ParetoEPCA
include("family/pareto.jl")

export
    WeibullEPCA
include("family/weibull.jl")

export
    EPCACompressor
include("compressor.jl")

#     HyperbolicSecantEPCA,
#     InverseGaussianEPCA,

end # module ExpFamilyPCA
