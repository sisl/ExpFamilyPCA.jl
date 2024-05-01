using Symbolics
using Optim


mutable struct ImplicitEPCA <: EPCA
    V
    G::Function
    g::Function
    Fg::Function
    fg::Function
end



function ImplicitEPCA(G::Function)
    # G induces g, Fg = F(g(θ)), and fg = f(g(θ))
    @variables θ
    D = Differential(θ)
    _g = expand_derivatives(D(G(θ)))
    _Fg = _g * θ - G(θ)
    _fg = expand_derivatives(D(_Fg))
    ex = quote
        g(θ) = $(Symbolics.toexpr(_g))
        Fg(θ) = $(Symbolics.toexpr(_Fg))
        fg(θ) = $(Symbolics.toexpr(_fg))
    end
    eval(ex)
    ImplicitEPCA(missing, G, g, Fg, fg)
end


function _binary_search_monotone(f, target; low=-1e10, high=1e10, tol=eps())
    while high - low > tol
        mid = (low + high) / 2
        if f(mid) < target
            low = mid
        else
            high = mid
        end
    end
    return (low + high) / 2
end


function _make_loss(epca::ImplicitEPCA, X, μ, ϵ; tol=eps())
    G, g, Fg, fg = epca.G, epca.g, epca.Fg, epca.fg
    g⁻¹X = map(x->_binary_search_monotone(g, x; tol=tol), X)
    g⁻¹μ = _binary_search_monotone(g, μ; tol=eps())  # NOTE: μ is scalar, so we can have very low tol
    FX = @. g⁻¹X * X - G(g⁻¹X)
    Fμ = @. g⁻¹μ * μ - G(g⁻¹μ)
    function bregman(θ)
        Fgθ = Fg.(θ)
        fgθ = fg.(θ)
        gθ = g.(θ)
        BFX = @. FX - Fgθ - fgθ * (X - gθ)
        BFμ = @. Fμ - Fgθ - fgθ * (μ - gθ)
        divergence = @. BFX - ϵ * BFμ
        return sum(divergence)W
    end
    return bregman
end


# TODO: change to StatsAPI?
# TODO: support "refitting"
function fit!(
    epca::ImplicitEPCA, 
    X,
    μ;
    maxoutdim=1, 
    maxiter=100,
    verbose=false,
    ϵ=eps(),
    tol=eps()
)
    L = _make_loss(epca, X, μ, ϵ; tol=tol)
    n, d = size(X)
    A = ones(n, maxoutdim)
    V = ones(maxoutdim, d)
    for _ in 1:maxiter
        V = Optim.minimizer(optimize(V̂->L(A * V̂), V))
        A = Optim.minimizer(optimize(Â->L(Â * V), A))
        if verbose
            @show L(A * V)
        end
    end
    epca.V = V
    return A
end

function compress(
    epca::ImplicitEPCA, 
    X,
    μ;
    maxiter=100,
    verbose=false,
    ϵ=eps(),
    tol=eps()
)
    L = _make_loss(epca, X, μ, ϵ; tol=tol)
    n, _ = size(X)
    A = ones(n, maxoutdim)
    V = epca.V
    for _ in 1:maxiter
        A = Optim.minimizer(optimize(Â->L(Â * V), A))
        if verbose
            @show L(A * V)
        end
    end
    return A
end

decompress(epca::ImplicitEPCA, A) = epca.g(A * epca.V)

    