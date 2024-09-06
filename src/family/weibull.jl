"""
"""
function WeibullEPCA(indim::Integer, outdim::Integer)
    Bg(x, θ) = -log(-θ) - x * θ
    g(θ) = -1 / θ
    epca = EPCA(
        indim,
        outdim,
        Bg,
        g,
        Val((:Bg, :g));
        options = Options(
            A_init_value = -1,
            A_upper = -eps(),
            V_lower = eps()
        )
    )
    return epca
end