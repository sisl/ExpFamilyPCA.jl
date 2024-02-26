using CompressedBeliefMDPs: Compressor, fit!, compress, decompress
using Symbolics
using NonlinearSolve


struct EPCA <: Compressor
    V::Matrix{<:Real}
    g::Function
    Bregman::Function
    μ0::Real
end


function EPCA(l::Integer, d::Integer, g::Function, Bregman::Function, μ0::Real)
    @assert 0 < l < d
    V = rand(l, d)
    return EPCA(V, g, Bregman, μ0)
end


function calc_inverse(g::Function, X)
    g_inv(u, p) = @. g(u) - p
    u0 = rand(size(X)...)
    prob = NonlinearProblem(g_inv, u0, X)
    sol = solve(prob)
    return sol.u
end


function EPCA(l::Integer, d::Integer, G::Num, μ0::Real)
    vars = Symbolics.get_variables(G)
    @assert length(vars) == 1 "G must be univariate"
    @assert size(vars[1]) == ()  "G must be ℝ → ℝ"

    # calculate F and f = F'
    θ = vars[1]
    D = Differential(θ)
    g = expand_derivatives(D(G))
    Fg = g * θ - G
    fg = D(Fg)

    # convert to callable Julia functions
    ex = quote
        _G(θ) = $(Symbolics.toexpr(G))
        _g(θ) = $(Symbolics.toexpr(g))
        _Fg(θ) = $(Symbolics.toexpr(Fg))
        _fg(θ) = $(Symbolics.toexpr(fg))
    end
    eval(ex)

    # build Bregman divergence
    _F(x) = @. _g(x) * calc_inverse(_g, x) - _G(x)
    Bregman(p, q) = @. _F(p) - _Fg(q) - _fg(q) * (p - q)
    
    return EPCA(l, d, x->_g.(x), Bregman, μ0)
end


function make_loss(epca::EPCA, X)
    L(A, V) = sum(epca.Bregman(X, epca.g(A * V)) + eps() * epca.Bregman(epca.μ0, epca.g(A * V)))
    return L
end

# TODO: maybe add type hinting for X from compressor.jl in BeliefCompression
function CompressedBeliefMDPs.fit!(epca::EPCA, X; verbose=false, maxiter::Integer=50)
    @assert maxiter > 0
    L(A, V) = make_loss(epca, X)
    n, _ = size(X)
    l, _ = size(epca.V)
    Â = zeros(n, l)
    V̂ = epca.V
    for _ in 1:maxiter
        if verbose println("Loss: ", L(Â, V̂)) end
        V̂ = Optim.minimizer(optimize(V->epca.L(Â, V), V̂))
        Â = Optim.minimizer(optimize(A->L(A, V̂), Â))
    end
    copyto!(epca.V, V̂)
end


function CompressedBeliefMDPs.compress(epca::EPCA, X; verbose=false, maxiter::Integer=50)
    @assert maxiter > 0
    n, _ = size(X)
    Â = zeros(n, size(epca.V)[1])
    L = make_loss(epca, X)
    for _ in 1:maxiter
        if verbose println("Loss: ", L(Â, epca.V)) end
        Â = Optim.minimizer(optimize(A->L(A, epca.V), Â))
    end
    X̃ = Â * epca.V
    return X̃
end


CompressedBeliefMDPs.decompress(epca::EPCA, X̃) = epca.g(X̃)
