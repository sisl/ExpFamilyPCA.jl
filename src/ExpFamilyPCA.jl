module ExpFamilyPCA

using Infiltrator

export
    EPCA,
    fit!,
    compress,
    decompress
include("epca.jl")

export
    ImplicitEPCA
include("implicit.jl")

export
    ExplicitEPCA,
    NormalEPCA,
    BernoulliEPCA,
    PoissonEPCA
include("explicit.jl")

end # module ExpFamilyPCA
