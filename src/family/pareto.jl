"""
m = minimum value
"""
function ParetoEPCA(
    indim::Integer, 
    outdim::Integer, 
    m::Real; 
    options::Options = Options(
        μ = 2,
        A_init_value = 2,
        A_lower = 1 / indim,
        V_init_value = -2,
        V_upper = -1,
    )
)
    @assert m > 0 "Minimum value m must be positive."
    @assert options.μ != log(m) "μ must be in the range of g."
    Bg(x, θ) = -log(-1 - θ) + xlogy(θ, m) - x * θ 
    g(θ) = log(m) - 1 / (θ + 1)
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = Options(
            μ = 2,
            A_init_value = 2,
            A_lower = 1 / indim,
            V_init_value = -2,
            V_upper = -1,
        )
    )
    return epca
end