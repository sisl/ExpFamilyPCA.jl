mutable struct ImplicitEPCA <: EPCA
    V
    G::Function
    g::Function
    Fg::Function
    fg::Function

    # hyperparameters
    tol
    mu
    epsilon
end


function EPCA(G::Function)
    return ImplicitEPCA(G::Function)
end


function ImplicitEPCA(G::Function; tol=eps(), mu=1, epsilon=eps())
    # NOTE: mu must be in the range of g, so g_inv(mu) is finite. It is up to the user to enforce this.
    # G induces g, Fg = F(g(theta)), and fg = f(g(theta))
    @variables theta
    D = Differential(theta)
    _g = expand_derivatives(D(G(theta)))
    _Fg = _g * theta - G(theta)
    _fg = expand_derivatives(D(_Fg))
    ex = quote
        g(theta) = $(Symbolics.toexpr(_g))
        Fg(theta) = $(Symbolics.toexpr(_Fg))
        fg(theta) = $(Symbolics.toexpr(_fg))
    end
    eval(ex)
    ImplicitEPCA(missing, G, g, Fg, fg, tol, mu, epsilon)
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
    G, g, Fg, fg, tol, mu, epsilon = epca.G, epca.g, epca.Fg, epca.fg, epca.tol, epca.mu, epca.epsilon
    g_inv_X = map(x->_binary_search_monotone(g, x; tol=tol), X)
    g_inv_mu = _binary_search_monotone(g, mu; tol=eps())  # NOTE: mu is scalar, so we can have very low tol
    F_X = @. g_inv_X * X - G(g_inv_X)
    F_mu = g_inv_mu * mu - G(g_inv_mu)
    function bregman(theta)
        @infiltrate
        Fg_theta = Fg.(theta)
        fg_theta = fg.(theta)
        g_theta = g.(theta)
        BF_X = @. F_X - Fg_theta - fg_theta * (X - g_theta)
        BF_mu = @. F_mu - Fg_theta - fg_theta * (mu - g_theta)
        divergence = @. BF_X - epsilon * BF_mu
        return sum(divergence)
    end
    return bregman
end