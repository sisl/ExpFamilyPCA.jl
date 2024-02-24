module ExpFamilyPCA

using Optim
using CompressedBeliefMDPs

# TODO: make this an imutable struct
mutable struct EPCA <: CompressedBeliefMDPs.Compressor
    n::Int  # number of samples
    d::Int  # size of each sample
    l::Int  # number of components
    A::Matrix  # n x l matrix
    V::Matrix  # l x d basis matrix

    G  # convex function that induces g, F, f, and Bregman
    g  # g(θ) = G'(θ)
    F  # F(g(θ)) + G(θ) = g(θ)θ
    f  # f(x) = F'(x)
    Bregman  # generalized Bregman divergence induced by F

    μ0::Real  # for numerical stability; must be in the range of g
    ϵ::Real  # controls weight of stabilizing term in loss function

    EPCA() = new()
end


# TODO: implement this with Symbolics of SymEnginer
# """
#     EPCA(G)

# Return the EPCA induced by a convex function G.
# """
# function EPCA(G)
#     return nothing
# end

# TODO: move this logic 
function EPCA(l::Int, μ0::Real; ϵ::Float64=0.01)
    epca = EPCA()
    epca.l = l
    epca.μ0 = μ0
    epca.ϵ = ϵ
    return epca
end


function CompressedBeliefMDPs.fit!(epca::EPCA, X; verbose=false, maxiter::Int=50)
    @assert epca.l > 0
    epca.n, epca.d = size(X)
    epca.A = zeros(epca.n, epca.l)
    epca.V = rand(epca.l, epca.d)

    L(A, V) = sum(epca.Bregman(X, epca.g(A * V)) + epca.ϵ * epca.Bregman(epca.μ0, epca.g(A * V)))

    for _ in 1:maxiter
        if verbose println("Loss: ", L(epca.A, epca.V)) end
        epca.V = Optim.minimizer(optimize(V->L(epca.A, V), epca.V))
        epca.A = Optim.minimizer(optimize(A->L(A, epca.V), epca.A))
    end
end

# TODO: make sure this works for both matrices and vectors!! also update the signature in compressed belief pomdps
function CompressedBeliefMDPs.compress(epca::EPCA, X; maxiter=50, verbose=false)
    n, d = size(X)
    @assert d == epca.d
    Â = zeros(n, epca.l)
    L(A, V) = sum(epca.Bregman(X, epca.g(A * V)) + epca.ϵ * epca.Bregman(epca.μ0, epca.g(A * V)))
    for _ in 1:maxiter
        if verbose println("Loss: ", L(Â, epca.V)) end
        Â = Optim.minimizer(optimize(A->L(A, epca.V), Â))
    end
    return Â * epca.V
end

CompressedBeliefMDPs.decompress(epca::EPCA, compressed) = epca.g(compressed)


export
    PoissonPCA
include("poisson.jl")

export
    BernoulliPCA
include("bernoulli.jl")


end # module ExpFamilyPCA
