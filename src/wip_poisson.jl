
function PoissonPCA()
    @. begin
        g(θ) = exp(θ)
        Bregman(p, q) = p * log((p + ϵ) / (q + ϵ)) + q - p  # with additive smoothing
    end
end