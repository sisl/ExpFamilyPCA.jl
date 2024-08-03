abstract type Objective <: Function end

struct EPCA{T} where T <: Real
    V::AbstractMatrix{T}
    L::Objective
    g::Function
end

function decompress(
    epca::EPCA{T}, 
    A::AbstractMatrix{T}
) where T <: Real
    Big_Theta = Θ
    small_theta = θ
    decompressed = epca
end
