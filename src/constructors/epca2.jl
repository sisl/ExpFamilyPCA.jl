struct EPCA2 <: EPCA
    V::AbstractMatrix{<:Real}
    G::Function
    g::Function  # link function
    Fg::Function  # Fg = F∘g (F compose g)
    fg::Function  # fg = f∘g (f compose g)

    # hyperparameters
    tol::Real
    μ::Real
    ϵ::Real
end

function _make_loss(epca::EPCA2, X)
    # unpack
    G = epca.G
    g = epca.g
    Fg = epca.Fg
    fg = epca.fg
    tol = epca.tol
    μ = epca.μ
    ϵ = epca.ϵ
    
    # construct EPCA objective
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
        loss = sum(divergence)
        return loss
    end
    return L
end

### CONSTRUCTORS ###


"""
TODO: add amonition about metaprogramming

Uses: G
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    ::Val{:G};
    tol=eps(),
    μ=1,
    ϵ=eps(),
    metaprogramming=true,
)
    # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
    # G induces g, Fg = F(g(θ)), and fg = f(g(θ))
    @variables θ
    D = Differential(θ)
    _g = expand_derivatives(D(G(θ)))
    _Fg = _g * θ - G(θ)  # By definition, F(g(θ)) + G(θ) = g(θ)⋅θ
    _fg = expand_derivatives(D(_Fg) / D(_g))  # Chain rule

    if metaprogramming
        g = _symbolics_to_julia(_g)
        Fg = _symbolics_to_julia(_Fg)
        fg = _symbolics_to_julia(_fg)
    else
        g = _symbolics_to_julia(_g, θ)
        Fg = _symbolics_to_julia(_Fg, θ)
        fg = _symbolics_to_julia(_fg, θ)
    end

    V = ones(outdim, indim)
    epca = EPCA2(
        V,
        G,
        g,
        Fg,
        fg,
        tol,
        μ,
        ϵ
    )
    return epca
end