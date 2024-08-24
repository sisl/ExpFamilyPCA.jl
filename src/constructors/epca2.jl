struct EPCA2 <: EPCA
    V::AbstractMatrix{<:Real}
    G::Function
    g::Function  # link function

    # hyperparameters
    tol::Real
    μ::Real
    ϵ::Real
end

# function _make_loss(epca::EPCA2, X)
#     # unpack
#     G = epca.G
#     g = epca.g
#     tol = epca.tol
#     μ = epca.μ
#     ϵ = epca.ϵ
    
#     # construct EPCA objective
#     Fg(θ) = g(θ) * θ - G(θ)  # By definition, F(g(θ)) + G(θ) = g(θ)⋅θ
#     g⁻¹X = map(X) do x
#         result = _binary_search_monotone(
#             g,
#             x; 
#             tol=tol
#         )
#         return result
#     end
#     g⁻¹μ = _binary_search_monotone(
#         g, 
#         μ; 
#         tol=0
#     )  # NOTE: μ is scalar, so we can have very low tol
#     FX = @. g⁻¹X * X - G(g⁻¹X)
#     Fμ = g⁻¹μ * μ - G(g⁻¹μ)
#     L(θ) = begin
#         gθ = g.(θ)  # think of this as X̂
#         Fgθ = Fg.(θ)  # Recall this is F(g(θ))
#         BF_X = @. FX - Fgθ - θ * (X - gθ)
#         BF_μ = @. Fμ - Fgθ - θ * (μ - gθ)
#         divergence = @. BF_X + ϵ * BF_μ
#         loss = sum(divergence)
#         return loss
#     end
# end

# function _make_loss(epca::EPCA2, X)
#     # unpack
#     G = epca.G
#     g = epca.g
#     tol = epca.tol
#     μ = epca.μ
#     ϵ = epca.ϵ

#     B_G(p, q) = G(p) - G(q) - g(q) * (p - q)
#     fX = _binary_search_monotone.(
#         g,
#         X; 
#         tol=tol
#     )
#     fμ = _binary_search_monotone(
#         g, 
#         μ; 
#         tol=0
#     )  # NOTE: μ is scalar, so we can have very low tol

#     L(θ) = begin
#         divergence = @. B_G(θ, fX)
#         regularizer = @. B_G(θ, fμ)
#         loss = sum(@. divergence + ϵ * regularizer)
#         return loss
#     end
# end

# function _make_loss(epca::EPCA2, X)
#     # unpack
#     G = epca.G
#     g = epca.g
#     tol = epca.tol
#     μ = epca.μ
#     ϵ = epca.ϵ

#     fX = _binary_search_monotone.(
#         g,
#         X; 
#         tol=tol
#     )
#     fXX = fX .* X
#     GfX = G.(fX)
    
#     fμ = _binary_search_monotone(
#         g, 
#         μ; 
#         tol=0
#     )  # NOTE: μ is scalar, so we can have very low tol
#     fμμ = fμ .* μ
#     Gfμ = G.(fμ)

#     L(θ) = begin
#         gθ = g.(θ)
#         z = @. gθ * θ - G(θ)
#         divergence = @. fXX - GfX - z - θ * (X - gθ)
#         regularizer = @. fμμ - Gfμ - z - θ * (μ - gθ)
#         loss = sum(@. divergence + ϵ * regularizer)
#         return loss
#     end
# end

function _make_loss(epca::EPCA2, X)
    # unpack
    G = epca.G
    g = epca.g
    tol = epca.tol
    μ = epca.μ
    ϵ = epca.ϵ

    fX = _binary_search_monotone.(
        g,
        X; 
        tol=tol
    )
    zX = @. fX * X - G(fX)
    
    fμ = _binary_search_monotone(
        g, 
        μ; 
        tol=0
    )  # NOTE: μ is scalar, so we can have very low tol
    zμ = @. fμ * μ - G(fμ)

    L(θ) = begin
        Gθ = G.(θ)
        divergence = @. zX + Gθ - θ * X
        regularizer = @. zμ + Gθ - θ * μ
        loss = sum(@. divergence + ϵ * regularizer)
        return loss
    end
end

function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    g::Function,
    ::Val{(:G, :g)};
    tol=eps(),
    μ=1,
    ϵ=eps(),
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert ϵ > 0 "ϵ must be positive."

    V = zeros(outdim, indim)
    epca = EPCA2(
        V,
        G,
        g,
        tol,
        μ,
        ϵ
    )
    return epca
end

"""
 # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    ::Val{(:G)};
    tol=eps(),
    μ=1,
    ϵ=eps(),
    metaprogramming=true,
)
    # assertions
    @assert indim > 0 "Input dimension (indim) must be a positive integer."
    @assert outdim > 0 "Output dimension (outdim) must be a positive integer."
    @assert indim >= outdim "Input dimension (indim) must be greater than or equal to output dimension (outdim)."
    @assert ϵ > 0 "ϵ must be positive."

    # NOTE: μ must be in the range of g, so g⁻¹(μ) is finite. It is up to the user to enforce this.
    # G induces g, Fg = F(g(θ)), and fg = f(g(θ))
    @variables θ
    D = Differential(θ)
    _g = expand_derivatives(D(G(θ)))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    epca = EPCA(
        indim,
        outdim,
        G,
        g,
        Val((:G, :g));
        tol=tol,
        μ=μ,
        ϵ=ϵ,
    )
    return epca
end