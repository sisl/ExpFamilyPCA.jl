function _single_fit_iter(L::Function, V, A, verbose::Bool, i::Integer, steps_per_print::Integer, maxiter::Integer)
    V = Optim.minimizer(optimize(V̂->L(A * V̂), V))
    A = _single_compress_iter(L, V, A, verbose, i, steps_per_print, maxiter)
    return V, A
end


function _single_compress_iter(L::Function, V, A, verbose::Bool, i::Integer, steps_per_print::Integer, maxiter::Integer)
    result = optimize(Â->L(Â * V), A)
    A = Optim.minimizer(result)
    if verbose && (i % steps_per_print == 0 || i == 1)
        loss = Optim.minimum(result)
        println("Iteration: $i/$maxiter | Loss: $loss")
    end
    return A
end


function _fit!(epca::EPCA, X, maxoutdim::Integer, L::Function, maxiter::Integer, verbose::Bool, steps_per_print::Integer, A_init)
    n, d = size(X)
    V = ismissing(epca.V) ? ones(maxoutdim, d) : epca.V
    @assert size(V) == (maxoutdim, d)
    A = isnothing(A_init) ? ones(n, maxoutdim) : A_init
    @assert size(A) == (n, maxoutdim)
    for i in 1:maxiter
        V, A = _single_fit_iter(L, V, A, verbose, i, steps_per_print, maxiter)
    end
    epca.V = V
    return A
end


function _compress(epca::EPCA, X, L::Function, maxiter::Integer, verbose::Bool, steps_per_print::Integer, A_init)
    n, _ = size(X)
    outdim = size(epca.V)[1]
    A = isnothing(A_init) ? ones(n, outdim) : A_init
    @assert size(A) == (n, outdim)
    for i in 1:maxiter
        A = _single_compress_iter(L, epca.V, A, verbose, i, steps_per_print, maxiter)
    end
    return A
end


function fit!(epca::EPCA, X; maxoutdim=1, maxiter=10, verbose=false, steps_per_print=10, A_init=nothing)
    L = _make_loss(epca, X)
    A =  _fit!(epca, X, maxoutdim, L, maxiter, verbose, steps_per_print, A_init)
    return A
end


function compress(epca::EPCA, X; maxiter=10, verbose=false, steps_per_print=10, A_init=nothing)
    L = _make_loss(epca, X)
    A = _compress(epca, X, L, maxiter, verbose, steps_per_print, A_init)
    return A
end


decompress(epca::EPCA, A) = epca.g(A * epca.V)