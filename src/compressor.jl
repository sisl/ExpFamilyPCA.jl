import CompressedBeliefMDPs  # `import` rather than `using` to keep tidey namespace and avoid collisions

"""
    EPCACompressor(epca::EPCA)

Compressor for `CompressedBeliefMDPs.jl`.
"""
struct EPCACompressor{E<:EPCA} <: CompressedBeliefMDPs.Compressor
    epca::E
end

function (c::EPCACompressor)(beliefs)
    if ndims(beliefs) == 2
        result = compress(c.epca, beliefs; maxiter=10)
    else
        result = vec(compress(c.epca, beliefs'; maxiter=10))
    end
    return result
end

function CompressedBeliefMDPs.fit!(c::EPCACompressor, beliefs)
    ExpFamilyPCA.fit!(c.epca, beliefs)
end
