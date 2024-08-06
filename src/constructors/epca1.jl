struct EPCA1{T} <: EPCA{T}
    V::AbstractMatrix{T}
    Bregman::Union{Function, PreMetric}  # Bregman divergence can be specified using a Julia Function or a Distances.jl PreMetric
    g::Function  # g function

    # hyperparameters
    μ::Real
    ϵ::Real
end

function _make_loss(epca::EPCA1, X)
    B = epca.Bregman
    g = epca.g
    μ = epca.μ
    ϵ = epca.ϵ
    L(θ) = begin
        X̂ = g.(θ)
        divergence = @. B(X, X̂) + ϵ * B(μ, X̂)
        return sum(divergence)
    end
    return L
end

### CONSTRUCTORS ### 

"""
Uses: Bregman, g
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    Bregman::Union{Function, PreMetric},
    g::Function; 
    μ=1,
    ϵ=eps()
)
    V = ones(outdim, indim)
    epca = EPCA1(
        V,
        Bregman,
        g,
        μ,
        ϵ
    )
    return epca
end

"""Uses: F, f, g"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    f::Function,
    g::Function; 
    μ=1,
    ϵ=eps()
)
    V = ones(outdim, indim)
    Bregman = Distances.Bregman(F, f)
    epca = EPCA1(
        V,
        Bregman,
        g,
        μ,
        ϵ
    )
    return epca
end

"""
TODO: add some adominition about how using metaprogramming polutes the global scope with eval 
Uses: F, f, G
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    f::Function,
    g::Function;
    μ=1,
    ϵ=eps(),
    use_metaprogramming=false
)
    V = ones(outdim, indim)
    Bregman = Distances.Bregman(F, f)

    # g = G'(θ)
    @variables θ
    G = G(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(G))
    if use_metaprogramming
        ex = quote
            g(θ) = $(Symbolics.to_expr(_g))
        end
        eval(ex)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    epca = ExplicitEPCA(
        V,
        Bregman,
        g,
        μ,
        ϵ
    )
    return epca
end

"""Uses: F, f"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    f::Function,
    g::Function; 
    μ=1,
    ϵ=eps()
)
    V = ones(outdim, indim)
    Bregman = Distances.Bregman(F, f)
    # TODO
    g = nothing
    epca = EPCA1(
        V,
        Bregman,
        g,
        μ,
        ϵ
    )
    return epca
end
