function PoissonPCA(l::Integer, d::Integer; μ0::Real=1)
    ϵ = eps()
    @. begin
        g(θ) = exp(θ)
        Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p  # with additive smoothing
    end
    epca = EPCA(l, d, g, Bregman, μ0)
    return epca
end


# TODO: include a normalized Poisson w/ link function in footnote 5 of long paper
function NormalizedPCA(l::Integer, d::Integer; μ0::Real=1)
    # TODO: handle possible problem w/ dimension of theta
    return
end