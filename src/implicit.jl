mutable struct ImplicitEPCA <: EPCA
    V
    G::Function
    g::Function
    Fg::Function
    fg::Function

    # hyperparameters
    tol
    μ
    ϵ
end


function EPCA(G::Function; tol=eps(), μ=1, ϵ=eps())
    return ImplicitEPCA(G::Function; tol=tol, μ=μ, ϵ=ϵ)
end


function ImplicitEPCA(G::Function; tol=eps(), μ=1, ϵ=eps())
    # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
    # G induces g, Fg = F(g(θ)), and fg = f(g(θ))
    @variables θ
    D = Differential(θ)
    _g = expand_derivatives(D(G(θ)))
    _Fg = _g * θ - G(θ)
    _fg = expand_derivatives(D(_Fg) / D(_g))
    ex = quote
        g(θ) = $(Symbolics.toexpr(_g))
        Fg(θ) = $(Symbolics.toexpr(_Fg))
        fg(θ) = $(Symbolics.toexpr(_fg))
    end
    eval(ex)
    ImplicitEPCA(missing, G, g, Fg, fg, tol, μ, ϵ)
end


function _binary_search_monotone(f, target; low=-1e10, high=1e10, tol=1e-10, maxiter=1e6)
    iter = 0
    while high - low > tol && iter < maxiter
        mid = (low + high) / 2
        if f(mid) < target
            low = mid
        else
            high = mid
        end
        iter += 1
    end
    return (low + high) / 2
end


function _make_loss(epca::ImplicitEPCA, X)
    G, g, Fg, fg, tol, μ, ϵ = epca.G, epca.g, epca.Fg, epca.fg, epca.tol, epca.μ, epca.ϵ
    g⁻¹X = map(x->_binary_search_monotone(g, x; tol=tol), X)
    g⁻¹μ = _binary_search_monotone(g, μ; tol=0)  # NOTE: μ is scalar, so we can have very low tol
    FX = @. g⁻¹X * X - G(g⁻¹X)
    Fμ = g⁻¹μ * μ - G(g⁻¹μ)
    L(θ) = begin
        X̂ = g.(θ)
        Fgθ = Fg.(θ)  # Recall this is F(g(θ))
        fgθ = fg.(θ)
        BF_X = @. FX - Fgθ - fgθ * (X - X̂)
        BF_μ = @. Fμ - Fgθ - fgθ * (μ - X̂)
        divergence = @. BF_X + ϵ * BF_μ
        return sum(divergence)
    end
    return L
end


function _make_loss_old(epca::ImplicitEPCA, X)
    G, g, Fg, fg, tol, μ, ϵ = epca.G, epca.g, epca.Fg, epca.fg, epca.tol, epca.μ, epca.ϵ
    g⁻¹X = map(x->_binary_search_monotone(g, x; tol=tol), X)
    g⁻¹μ = _binary_search_monotone(g, μ; tol=0)  # NOTE: μ is scalar, so we can have very low tol
    FX = @. X * g⁻¹X - G(g⁻¹X)
    Fμ = μ * g⁻¹μ - G(g⁻¹μ)
    L(θ) = begin
        @infiltrate
        X̂  = g.(θ)
        B1 = @. FX - Fg(θ) - fg(θ) * (X - X̂)
        B2 = @. Fμ - Fg(θ) - fg(θ) * (μ - X̂)
        divergence = @. B1 + ϵ * B2
        return sum(divergence)
    end
    return L
end