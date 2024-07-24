struct HybridEPCA <: EPCA
    V
    Bregman  # Bregman divergence
    f

    # hyperparameters
    tol
    μ
    ϵ
end


"""Induce the Bregman divergence from F."""
function EPCA(indim, outdim, F::Function; tol=eps(), μ=1, ϵ=eps())
    @variables θ
    D = Differential(θ)
    _f = expand_derivatives(D(F(θ)))
    ex = quote
        f(θ) = $(Symbolics.toexpr(_f))
    end
    eval(ex)
    Bregman = Distances.Bregman(F, f)
    V = ones(outdim, indim)
    epca = HybridEPCA(
        V,
        Bregman,
        f,
        tol,
        μ,
        ϵ
    )
    return epca
end