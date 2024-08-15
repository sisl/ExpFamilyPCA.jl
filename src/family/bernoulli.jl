struct BernoulliEPCA <: EPCA
    V::AbstractMatrix{<:Real}
    g::Function
end

function _make_loss(::BernoulliEPCA, X)
    L(θ) = begin
        xp = @. 2X - 1
        z = @. exp(-xp * θ)
        divergence = log1p(z)
        return sum(divergence)
    end
    return L
end