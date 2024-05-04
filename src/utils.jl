function _single_fit_iter(L::Function, V, A, verbose, i, steps_per_print)
    V = Optim.minimizer(optimize(V_hat->L(A * V_hat), V))
    A = _single_compress_iter(L, V, A, verbose, i, steps_per_print)
    return V, A
end

function _single_compress_iter(L::Function, V, A, verbose, i, steps_per_print)
    result = optimize(A_hat->L(A_hat * V), A)
    A = Optim.minimizer(result)
    if verbose && (i % steps_per_print == 0 || i == 1)
        loss = Optim.minimum(result)
        println("Iteration: $i/$maxiter | Loss: $loss")
    end
    return A
end

function _fit!(epca::EPCA, X, maxoutdim, L, verbose, steps_per_print, maxiter)
    n, d = size(X)
    A = ones(n, maxoutdim)
    V = ismissing(epca.V) ? ones(maxoutdim, d) : epca.V
    for i in 1:maxiter
        V, A = _single_fit_iter(L, V, A, verbose, i, steps_per_print)
    end
    epca.V = V
    return A
end

function _compress(epca::EPCA, X, L, maxiter, verbose, steps_per_print)
    V = epca.V
    n, _ = size(X)
    outdim = size(V)[1]
    A = ones(n, outdim)
    for i in 1:maxiter
        _single_compress_iter(L, V, A, verbose, i, steps_per_print)
    end
    return A
end

decompress(epca::EPCA, A) = epca.g(A * epca.V)
