using Symbolics
using StatsAPI


struct EPCA <: StatsAPI.StatisticalModel
    V::Matrix
    g::function
    F::function
    f::function
end

function EPCA(G::Function)
    # G induces g, F(g(θ)), and f(g(θ))
    @variables θ
    D = Differential(θ)
    begin
        local g = expand_derivatives(D(G(θ)))
        local Fg = g * θ - G(θ)
        local fg = expand_derivatives(D(Fg))
        ex = quote
            g(θ) = $(Symbolics.toexpr(g))
            Fg(θ) = $(Symbolics.toexpr(Fg))
            fg(θ) = $(Symbolics.toexpr(fg))
        end
    end
    eval(ex)
end

function _make_loss(epca::EPCA, X)
    F

function fit!(epca::EPCA, X; l=1, maxiter=50, verbose=false)
    A = ones(n, maxoutdim)
    V = ones(maxoutdim)
    for _ in 1:maxiter
        V = Optim.minimizer(optimize(V̂->L(A, V̂), V))
        A = Optim.minimizer(optimize(Â->L(Â, V), A))
    end
end

    