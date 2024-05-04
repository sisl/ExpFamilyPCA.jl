using Symbolics
using Optim


mutable struct ImplicitEPCA <: EPCA
    V
    G::Function
    g::Function
    Fg::Function
    fg::Function
end

function EPCA(G::Function)
    return ImplicitEPCA(G::Function)
end


function ImplicitEPCA(G::Function)
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
    ImplicitEPCA(missing, G, g, Fg, fg)
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


function _make_loss(epca::ImplicitEPCA, X, mu, epsilon; tol=eps())
    G, g, Fg, fg = epca.G, epca.g, epca.Fg, epca.fg
    g_inv_X = map(x->_binary_search_monotone(g, x; tol=tol), X)
    g_inv_mu = _binary_search_monotone(g, mu; tol=eps())  # NOTE: mu is scalar, so we can have very low tol
    F_X = @. g_inv_X * X - G(g_inv_X)
    F_mu = @. g_inv_mu * mu - G(g_inv_mu)
    function bregman(theta)
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


function fit!(
    epca::ImplicitEPCA, 
    X,
    mu;
    maxoutdim=1, 
    maxiter=100,
    verbose=false,
    print_steps=10,
    epsilon=eps(),
    tol=eps()
)
    L = _make_loss(epca, X, mu, epsilon; tol=tol)
    n, d = size(X)
    A = ones(n, maxoutdim)
    V = ismissing(epca.V) ? ones(maxoutdim, d) : epca.V
    for i in 1:maxiter
        V = Optim.minimizer(optimize(V_hat->L(A * V_hat), V))
        result = optimize(A_hat->L(A_hat * V), A)
        A = Optim.minimizer(result)
        if verbose && (i % print_steps == 0 || i == 1)
            loss = Optim.minimum(result)
            println("Iteration: $i/$maxiter | Loss: $loss")
        end
    end
    epca.V = V
    return A
end

function compress(
    epca::ImplicitEPCA, 
    X;
    mu=1,  # NOTE: mu = 1 may not be valid for all link functions. 
    maxiter=100,
    verbose=false,
    print_steps=10,
    epsilon=eps(),
    tol=eps()
)
    L = _make_loss(epca, X, mu, epsilon; tol=tol)
    n, _ = size(X)
    V = epca.V
    outdim = size(V)[1]
    A = ones(n, outdim)
    for i in 1:maxiter
        result = optimize(A_hat->L(A_hat * V), A)
        A = Optim.minimizer(result)
        if verbose && (i % print_steps == 0 || i == 1)
            loss = Optim.minimum(result)
            println("Iteration: $i/$maxiter | Loss: $loss")
        end
    end
    return A
end

decompress(epca::ImplicitEPCA, A) = epca.g(A * epca.V)

    