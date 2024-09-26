using Infiltrator
using Revise

using ExpFamilyPCA
using CompressedBeliefMDPs
using Random
using Plots
using Statistics

Random.seed!(1)

function kl_div(p, q; ϵ=eps())
    return sum(@. p * log((p + ϵ) / (q + ϵ)))
end

function to_probability(matrix::AbstractMatrix)
    matrix = clamp.(matrix, 0, 1)
    prob_matrix = matrix ./ sum(matrix, dims=2)
    return prob_matrix
end

function calc_kl(
    epca::EPCA, 
    X::AbstractMatrix{T}
) where T<:Real
    V = epca.V
    A = ExpFamilyPCA.fit!(epca, X; maxiter=100, verbose=true, steps_per_print=10)
    X_recon = decompress(epca, A)
    Q = to_probability(X_recon)
    divergences = []
    for (p, q) in zip(eachrow(X), eachrow(Q))
        kl = kl_div(p, q)
        push!(divergences, kl)
    end
    result = mean(divergences)  # Return the mean KL divergence
    @show result
    return result
end

n_corridors = 2
corridor_length = 20
maze = CircularMaze(n_corridors, corridor_length)
sampler = PolicySampler(maze, n=500)
raw_beliefs = sampler(maze)
beliefs = make_numerical(raw_beliefs, maze)

n, indim = size(beliefs)

outdims = 1:6

kl_divs_poisson_epca = []
kl_divs_gaussian_epca = []

for outdim in outdims
    @show outdim
    push!(kl_divs_poisson_epca, calc_kl(PoissonEPCA(indim, outdim), beliefs))
    push!(kl_divs_gaussian_epca, calc_kl(GaussianEPCA(indim, outdim), beliefs))
end

# Plotting
plot(
    outdims, 
    kl_divs_poisson_epca, 
    label="Poisson EPCA", 
    yscale=:log10,
    marker=:cross, 
    linestyle=:solid, 
    lw=2,
    dpi=600
)
plot!(
    outdims, 
    kl_divs_gaussian_epca, 
    label="PCA",
    yscale=:log10,
    marker=:x,
    linestyle=:dash,
    lw=2,
    dpi=600
)
xlabel!("Number of Bases")
ylabel!("KL Divergence")

# Save the plot
savefig("kl_divergence_plot.png")
