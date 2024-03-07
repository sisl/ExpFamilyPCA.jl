function GaussianPCA(l::Integer, d::Integer; μ0::Real=0)
    @. begin
        g(θ) = θ
        Bregman(p, q) = (p - q)^2 / 2
    end
    epca = EPCA(l, d, g, Bregman, μ0)
    return epca
end