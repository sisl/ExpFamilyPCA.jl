"""
r = number of failures
"""
function NegativeBinomialEPCA(
    indim::Integer, 
    outdim::Integer, 
    r::Integer;
    options::Options = Options(
        A_init_value = -1,
        A_upper = -eps(),
        V_lower = eps()
    )
)
    @assert r > 0 "Number of failures r must be positive."
    Bg(x, θ) = -r * log1mexp(θ) - x * θ
    g(θ) = xexpy(-r, θ) / expm1(θ)
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = options
    )
    return epca
end