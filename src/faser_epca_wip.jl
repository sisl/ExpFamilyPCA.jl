# using CompressedBeliefMDPs
# using Optim
# using Symbolics
# using NonlinearSolve


# struct EPCA <: Compressor
#     V::Matrix{<:Real}
#     g::Function
#     L::Function
#     μ0::Real
# end


# function EPCA(l::Integer, d::Integer, g::Function, L::Function, μ0::Real)
#     @assert 0 < l ≤ d
#     V = rand(l, d)
#     return EPCA(V, g, L, μ0)
# end


# function calc_inverse(g::Function, X)
#     # TEMP:
#     println("inverse called! this is slowing us down!")

#     g_inv(u, p) = @. g(u) - p
#     u0 = rand(size(X)...)
#     prob = NonlinearProblem(g_inv, u0, X)
#     sol = NonlinearSolve.solve(prob)
#     return sol.u
# end


# function EPCA(l::Integer, d::Integer, G::Function, μ0::Real)
#     @variables θ
#     G = G(θ)
#     D = Differential(θ)
#     g = expand_derivatives(D(G))
#     Fg = g * θ - G
#     fg = expand_derivatives(D(Fg))

#     # convert to callable Julia functions
#     ex = quote
#         _G(θ) = $(Symbolics.toexpr(G))
#         _g(θ) = $(Symbolics.toexpr(g))
#         _Fg(θ) = $(Symbolics.toexpr(Fg))
#         _fg(θ) = $(Symbolics.toexpr(fg))
#     end
#     eval(ex)

#     # build objective function
#     Fμ = @. _g(μ0) * calc_inverse(_g, μ0) - _G(μ0)
    
#     function L(A, V, Fx)  # TODO: perhaps make Fx an argument
#         Fx = @. _g(X) * calc_inverse(_g, X) - _G(X)  # TODO: move this outside of function too
#         θ = _g.(A * V)
#         Fθ = _Fg.(θ)
#         fθ = _fg.(θ)
#         breg1 = @. Fx - Fθ - fθ * (X - θ)
#         breg2 = @. Fμ - Fθ - fθ * (μ0 - θ)
#         return sum(breg1 + eps() * breg2)
#     end

#     return EPCA(l, d, x->_g.(x), L, μ0)
# end

# function make_loss(epca::EPCA, X)
#     L(A, V) = sum(epca.Bregman(X, epca.g(A * V)) + eps() * epca.Bregman(epca.μ0, epca.g(A * V)))
#     return L
# end


# # TODO: maybe add type hinting for X from compressor.jl in BeliefCompression
# # TODO: perhaps add early exit for some ϵ
# # TODO: make sure printing happens on 1 line
# function CompressedBeliefMDPs.fit!(epca::EPCA, X; verbose=false, maxiter::Integer=50)
#     @assert maxiter > 0
#     n, _ = size(X)
#     l, _ = size(epca.V)
#     Â = rand(n, l)
#     V̂ = epca.V
#     L = make_loss(epca, X)
#     for _ in 1:maxiter
#         V̂ = Optim.minimizer(optimize(V->epca.L(Â, V), V̂))
#         Â = Optim.minimizer(optimize(A->epca.L(A, V̂), Â))
#         if verbose println(L(Â, V̂)) end
#     end
#     copyto!(epca.V, V̂)
#     return
# end


# function CompressedBeliefMDPs.compress(epca::EPCA, X; verbose=false, maxiter::Integer=50)
#     @assert maxiter > 0
#     n, _ = size(X)
#     Â = rand(n, size(epca.V)[1])
#     L = make_loss(epca, X)
#     for _ in 1:maxiter
#         Â = Optim.minimizer(optimize(A->epca.L(A, epca.V), Â))
#         if verbose println(epca.L(Â, epca.V)) end
#     end
#     return Â
# end


# CompressedBeliefMDPs.decompress(epca::EPCA, A) = epca.g(A * epca.V)
