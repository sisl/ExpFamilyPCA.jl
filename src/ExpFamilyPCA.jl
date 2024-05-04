module ExpFamilyPCA

using Infiltrator

using Symbolics
using Optim

export
    EPCA,
    fit!,
    compress,
    decompress
include("epca.jl")

include("utils.jl")
include("implicit.jl")

export
    NormalEPCA,
    BernoulliEPCA,
    PoissonEPCA
include("explicit.jl")

end # module ExpFamilyPCA
