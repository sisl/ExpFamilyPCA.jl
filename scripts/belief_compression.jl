using ExpFamilyPCA
using CompressedBeliefMDPs
using POMDPs, POMDPTools

using JSON
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
    return result, divergences
end

n_corridors = 2
corridor_length = 20
maze = CircularMaze(n_corridors, corridor_length)

rng = MersenneTwister(100)
policy = RandomPolicy(maze; rng=rng)
sampler = PolicySampler(maze; policy=policy, rng=rng, n=100)

raw_beliefs = sampler(maze)
beliefs = make_numerical(raw_beliefs, maze)

n, indim = size(beliefs)

outdims = 1:7

kl_divs_poisson_epca = []
kl_divs_gaussian_epca = []
epca_data = Dict{Integer, Vector}()
pca_data = Dict{Integer, Vector}()

for outdim in outdims
    @show outdim
    epca_kl, epca_divergences = calc_kl(PoissonEPCA(indim, outdim), beliefs)
    pca_kl, pca_divergences = calc_kl(GaussianEPCA(indim, outdim), beliefs)
    epca_data[outdim] = epca_divergences
    pca_data[outdim] = pca_divergences
    push!(kl_divs_poisson_epca, epca_kl)
    push!(kl_divs_gaussian_epca, pca_kl)
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
title!("KL Divergence Across Bases", fontsize=14)
plot!(
    legendfontsize = 10,
    xtickfontsize = 10,
    ytickfontsize = 10,
    framestyle = :box,
    markerstrokewidth = 2,  # Makes markers more prominent
    markeralpha = 0.7  # Adds transparency to markers
)
xlabel!("Number of Bases", fontsize=12)
ylabel!("Average KL Divergence", fontsize=12)

# Save the plot and data
open("epca_data.json", "w") do file
    JSON.print(file, epca_data)
end
open("pca_data.json", "w") do file
    JSON.print(file, pca_data)
end
savefig("kl_divergence_plot.png")

