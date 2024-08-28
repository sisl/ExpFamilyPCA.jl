# TODO: add checking for μ to make sure in proper range

function PoissonEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps()
)
    # assumes χ = ℕ
    F(x) = x * log(x + ϵ) - x
    g(θ) = exp(θ)
    @assert μ > 0 "For PoissonEPCA, μ must be positive to be in the range of g(θ) = exp(θ)."
    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g)); 
        μ = μ,
        ϵ = ϵ
    )
    return epca
end

