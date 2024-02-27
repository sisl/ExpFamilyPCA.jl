using CompressedBeliefMDPs
using Optim
using Symbolics
using NonlinearSolve


struct EPCA <: Compressor
    V::Matrix{<:Real}
    g::Function
    Bregman::Function
    μ0::Real
end


function EPCA(l::Integer, d::Integer, g::Function, Bregman::Function, μ0::Real)
    @assert 0 < l ≤ d
    V = rand(l, d)
    # TODO: change this to accept the loss function instead --> this allows faster 
    return EPCA(V, g, Bregman, μ0)
end


function calc_inverse(g::Function, X)
    # TODO: remove debugging code
    # println("inverse called! this is slowing us down!")

    g_inv(u, p) = @. g(u) - p
    u0 = rand(size(X)...)
    prob = NonlinearProblem(g_inv, u0, X)
    sol = NonlinearSolve.solve(prob)
    return sol.u
end


function EPCA(l::Integer, d::Integer, G::Function, μ0::Real)
    @variables θ
    G = G(θ)
    D = Differential(θ)
    g = expand_derivatives(D(G))
    Fg = g * θ - G
    fg = expand_derivatives(D(Fg))

    # convert to callable Julia functions
    ex = quote
        _G(θ) = $(Symbolics.toexpr(G))
        _g(θ) = $(Symbolics.toexpr(g))
        _Fg(θ) = $(Symbolics.toexpr(Fg))
        _fg(θ) = $(Symbolics.toexpr(fg))
    end
    eval(ex)

    # build Bregman divergence
    # TODO: move this outside so that this only has to be calculated once
    _F(x) = @. _g(x) * calc_inverse(_g, x) - _G(x)
    Bregman(p, q) = @. _F(p) - _Fg(q) - _fg(q) * (p - q)

    return EPCA(l, d, x->_g.(x), Bregman, μ0)
end

function make_loss(epca::EPCA, X)
    L(A, V) = sum(epca.Bregman(X, epca.g(A * V)) + eps() * epca.Bregman(epca.μ0, epca.g(A * V)))
    return L
end


# TODO: maybe add type hinting for X from compressor.jl in BeliefCompression
# TODO: perhaps add early exit for some ϵ
# TODO: make sure printing happens on 1 line
function CompressedBeliefMDPs.fit!(epca::EPCA, X; verbose=false, maxiter::Integer=50)
    @assert maxiter > 0
    n, _ = size(X)
    l, _ = size(epca.V)
    Â = rand(n, l)
    V̂ = epca.V
    L = make_loss(epca, X)
    for _ in 1:maxiter
        V̂ = Optim.minimizer(optimize(V->L(Â, V), V̂))
        Â = Optim.minimizer(optimize(A->L(A, V̂), Â))
        if verbose println(L(Â, V̂)) end
    end
    copyto!(epca.V, V̂)
    return
end


function CompressedBeliefMDPs.compress(epca::EPCA, X; verbose=false, maxiter::Integer=50)
    @assert maxiter > 0
    n, _ = size(X)
    Â = rand(n, size(epca.V)[1])
    L = make_loss(epca, X)
    for _ in 1:maxiter
        Â = Optim.minimizer(optimize(A->L(A, epca.V), Â))
        if verbose println(L(Â, epca.V)) end
    end
    return Â
end


CompressedBeliefMDPs.decompress(epca::EPCA, A) = epca.g(A * epca.V)
