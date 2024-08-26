abstract type EPCA end


function fit! end
function compress end
function decompress end

function fit!(
    epca::EPCA,
    X::AbstractMatrix{T};
    maxiter::Integer = 100,
    verbose::Bool = false,
    steps_per_print::Integer = 10,
) where T <: Real
    L = _make_loss(epca, X)
    V = epca.V
    A = _initialize_A(epca, X)
    A_lower = epca.A_lower
    A_upper = epca.A_upper
    V_lower = epca.V_lower
    V_upper = epca.V_upper
    V, A = _fit(
        L,
        V,
        A,
        maxiter,
        verbose,
        steps_per_print;
        A_lower = A_lower,
        A_upper = A_upper,
        V_lower = V_lower,
        V_upper = V_upper
    )
    epca.V[:] = V
    return A
end

function compress(
    epca::EPCA,
    X::AbstractMatrix{T};
    maxiter::Integer = 100,
    verbose::Bool = false,
    steps_per_print::Integer = 10
) where T <: Real
    L = _make_loss(epca, X)
    V = epca.V
    A = _initialize_A(epca, X)
    A_lower = epca.A_lower
    A_upper = epca.A_upper
    A = _compress(
        L,
        V,
        A,
        maxiter,
        verbose,
        steps_per_print;
        A_lower = A_lower,
        A_upper = A_upper,
    )
    return A
end

function decompress(
    epca::EPCA,
    A::AbstractMatrix{T}
) where T <: Real
    natural_params = A * epca.V
    X̂ = epca.g.(natural_params)
    return X̂
end
