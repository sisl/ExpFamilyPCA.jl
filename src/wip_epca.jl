using CompressedBeliefMDPs
using Optim
using Symbolics
using NonlinearSolve


struct EPCA <: Compressor
    V::Matrix{<:Real}  # basis matrix
    G::Function
    g::Function  # link function
    L::Function  # loss function
    μ0::Real
end


function EPCA(l::Integer, d::Integer, G::Function, g::Function, L::Function, μ0::Real)
    @assert 0 < l ≤ d
    V = zeros(l, d)
    return EPCA(V, G, g, L, μ0)
end


function calc_inverse(g::Function, X)
    # TODO: remove TEMP
    # TODO: perhaps replace w/ a binary search to make use of g's monotonicity
    println("inverse called! this is slowing us down!")

    g_inv(u, p) = @. g(u) - p
    problem = NonlinearProblem(g_inv, zeros(size(X)...), X)
    solution = NonlinearSolve.solve(probem)
    return solution.u
end


function find_functions(G::Function)
    @variables θ
    G = G(θ)
    D = Differential(θ)
    g = expand_derivatives(D(G))
    Fg = g * θ - G
    fg = expand_derivatives(D(Fg))

    # convert to callable Julia functions
    ex = quote
        # TODO: replace w/ G = θ->$(Symbolics.toexpr(G)) to trick compiler
        _G(θ) = $(Symbolics.toexpr(G))
        _g(θ) = $(Symbolics.toexpr(g))
        _Fg(θ) = $(Symbolics.toexpr(Fg))
        _fg(θ) = $(Symbolics.toexpr(fg))
    end
    eval(ex)
    return _g, _Fg, _fg
end


function EPCA(l::Integer, d::Integer, G::Function, μ0::Real)
    (g, Fg, fg) = find_functions(G)

    # build objective function; precompute values
    Fμ = @. g(μ0) * calc_inverse(g, μ0) - G(μ0)
    function L(A, V, Fx, X)
        θ = g.(A * V)
        Fθ = Fg.(θ)
        fθ = fg.(θ)
        breg1 = @. Fx - Fθ - fθ * (X - θ)
        breg2 = @. Fμ - Fθ - fθ * (μ0 - θ)
        return sum(breg1 + eps() * breg2)
    end

    return EPCA(l, d, x->G.(x), x->g.(x), L, μ0)
end


function make_loss(epca::EPCA, X)
    Fx = @. epca.g(X) * calc_inverse(epca.g, X) - epca.G(X) 
    L(A, V) = epca.L(A, V, Fx, X)
    return L
end


# TODO: maybe add type hinting for X from compressor.jl in BeliefCompression
# TODO: perhaps add early exit for some ϵ
# TODO: make sure printing happens on 1 line
function CompressedBeliefMDPs.fit!(epca::EPCA, X; verbose=false, maxiter::Integer=50)
    @assert maxiter > 0
    n, _ = size(X)
    l, _ = size(epca.V)
    Â = zeros(n, l)
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
    Â = zeros(n, size(epca.V)[1])
    L = make_loss(epca, X)
    for _ in 1:maxiter
        Â = Optim.minimizer(optimize(A->L(A, epca.V), Â))
        if verbose println(L(Â, epca.V)) end
    end
    return Â
end


CompressedBeliefMDPs.decompress(epca::EPCA, A) = epca.g(A * epca.V)
