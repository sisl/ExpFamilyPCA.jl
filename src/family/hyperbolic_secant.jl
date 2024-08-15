# NOTE: default n and p values were chosen to match https://juliastats.org/Distributions.jl/v0.14/univariate.html#Distributions.Binomial

# TODO: redo math 

function HyperbolicSecantEPCA(
    indim::Integer,
    outdim::Integer;
    r=1,
    p=0.5,
    ϵ=eps()
)
    # TODO: χ = ???
    @. begin
        G(θ) = -r * log(cos(θ))
    end
    μ = 1
    epca = EPCA(
        indim,
        outdim,
        G,
        Val((:G));
        μ=μ,
        ϵ=ϵ
    )
    return epca
end