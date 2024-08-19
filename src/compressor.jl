import CompressedBeliefMDPs  # `import` rather than `using` to keep tidey namespace and avoid collisions


struct EPCACompressor <: CompressedBeliefMDPs.Compressor
    epca::EPCA
end

# TODO: make parameters part of constructor
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