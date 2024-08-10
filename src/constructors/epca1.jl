struct EPCA1 <: EPCA
    V::AbstractMatrix{<:Real}
    Bregman::Union{Function, PreMetric}  # Bregman divergence can be specified using a Julia Function or a Distances.jl PreMetric
    g::Function  # g function

    # hyperparameters
    μ::Real
    ϵ::Real
end


# TODO: make epca3 that stores F so we don't have to recompute F over and over
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
Uses: Bregman, G

# NOTE: here the Bregman must be implemented as a Julia function, not as a Distances.PreMetric b/c we need to differentiate G
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    Bregman::Function,
    ::Val{(:Bregman, :G)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true
)

    @variables θ
    G = G(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(G))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

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

"""
Uses: Bregman, g
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    Bregman::Union{Function, PreMetric},
    g::Function,
    ::Val{(:Bregman, :g)};
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
    g::Function,
    ::Val{(:F, :f, :g)}; 
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
TODO: add some adominition about how using metaprogramming polutes the global scope with eval --> don't know if this is true anymore
Uses: F, f, G
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    f::Function,
    G::Function,  
    ::Val{(:F, :f, :G)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true
)
    Bregman = Distances.Bregman(F, f)

    # g = G'(θ)
    @variables θ
    G = G(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(G))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    V = ones(outdim, indim)
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
    ::Val{(:F, :f)};
    μ=1,
    ϵ=eps(),
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)
    Bregman = Distances.Bregman(F, f)
    g = _invert_legrende(
        f;
        low=low,
        high=high,
        tol=tol,
        maxiter=maxiter
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

"""Uses: F"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    F::Function,
    ::Val{(:F)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true,
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)

    @variables θ
    F = F(θ)
    D = Differential(θ)
    _f = expand_derivatives(D(F))

    if metaprogramming
        f = _symbolics_to_julia(_f)
    else
        f = _symbolics_to_julia(_f, θ)
    end
    Bregman = Distances.Bregman(F, f)

    g = _invert_legrende(
        f;
        low=low,
        high=high,
        tol=tol,
        maxiter=maxiter
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

"""Uses: G, F

G - > g
F -> f

We never use inversion if we can use a derivative, b/c inversion uses binary search which is slower
"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    F::Function,
    ::Val{(:G, :F)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true
)

    @variables θ
    G = G(θ)
    F = F(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(G))
    _f = expand_derivatives(D(F))

    if metaprogramming
        g = _symbolics_to_julia(_g)
        f = _symbolics_to_julia(_f)
    else
        g = _symbolics_to_julia(_g, θ)
        f = _symbolics_to_julia(_f, θ)
    end
    Bregman = Distances.Bregman(F, f)

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


"""Uses: F, g"""
function EPCA(
    indim::Integer,
    outdim::Integer,
    g::Function,
    F::Function,
    ::Val{(:g, :F)};
    μ=1,
    ϵ=eps(),
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)
    f = _invert_legrende(
        g;
        low=low,
        high=high,
        tol=tol,
        maxiter=maxiter
    )
    Bregman = Distances.Bregman(F, f)
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


# TODO: potentially remove
function EPCA(
    indim::Integer,
    outdim::Integer,
    G::Function,
    ::Val{(:G1)};
    μ=1,
    ϵ=eps(),
    metaprogramming=true,
    low=-1e10, 
    high=1e10, 
    tol=1e-10, 
    maxiter=1e6
)

    @variables θ
    _G = G(θ)
    D = Differential(θ)
    _g = expand_derivatives(D(_G))

    if metaprogramming
        g = _symbolics_to_julia(_g)
    else
        g = _symbolics_to_julia(_g, θ)
    end

    f = _invert_legrende(
        g;
        low=low,
        high=high,
        tol=tol,
        maxiter=maxiter
    )
    F(ω) = f(ω) * ω - G(f(ω))

    Bregman = Distances.Bregman(F, f)

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