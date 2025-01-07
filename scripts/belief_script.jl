using Statistics
using StableRNGs
using JSON

using Plots
using POMDPs
using POMDPTools
using CompressedBeliefMDPs

using ExpFamilyPCA

rng = StableRNG(123)
logocolors = Colors.JULIA_LOGO_COLORS

function is_terminal_belief(belief)
    return all(x -> x == 0, belief[1:end-1]) && belief[end] == 1
end

function drop_terminal_beliefs(B)
    B_numerical = make_numerical(B, pomdp)
    B_numerical = filter(row -> !is_terminal_belief(row), eachrow(B_numerical))  # exclude belief in terminal state
    B_numerical = reduce(hcat, B_numerical)'
    B_numerical = B_numerical[:, 1:end - 1]
    return B_numerical
end

pomdp = CircularMaze(2, 100)
policy = RandomPolicy(pomdp; rng = rng)
sampler = PolicySampler(pomdp; policy = policy, rng = rng, n = 40)
beliefs = drop_terminal_beliefs(sampler(pomdp))

x = 1:length(states(pomdp)) - 1
b1 = beliefs[1, :]
n, indim = size(beliefs)
outdim = 10

@show size(beliefs)
plot(x, b1, linestyle=:solid, linewidth=2, legend=false)
xlabel!("State")
ylabel!("Probability")
title!("Initial Belief State")
# savefig("ExpFamilyPCA/scripts/init_belief.png")
savefig("init_belief.png")

pca = GaussianEPCA(indim, outdim)
A1 = ExpFamilyPCA.fit!(pca, beliefs; verbose=true)

recon1 = decompress(pca, A1)
plot(x, b1, 
     linestyle=:solid, 
     linewidth=1, 
     color=logocolors.red,
     label="Original Belief"
)

plot!(x, recon1[1, :], 
     linestyle=:dash, 
     linewidth=3, 
     color=logocolors.blue,
     label="Reconstructed Belief"
)

xlabel!("State", fontsize=12)
ylabel!("Probability", fontsize=12)
title!("PCA Reconstruction", fontsize=14)

xlims!(0, 200)
ylims!(0.003, 0.008)

plot!(legend=:topright, legendfontsize=10, framestyle=:box)
p1 = plot!(xtickfontsize=10, ytickfontsize=10)
# savefig("ExpFamilyPCA/scripts/PCA_recon.png")
savefig("PCA_recon.png")


epca = PoissonEPCA(indim, outdim)
A2 = ExpFamilyPCA.fit!(epca, beliefs; verbose=true)

recon2 = decompress(epca, A2)
plot(x, b1, 
     linestyle=:solid, 
     linewidth=1, 
     color=logocolors.red,
     label="Original",
     dpi=600
)

plot!(x, recon1[1, :], 
      linestyle=:solid, 
      linewidth=2, 
      color=logocolors.green,
      label="PCA",
      dpi=600
)

plot!(x, recon2[1, :], 
      linestyle=:dash, 
      linewidth=3, 
      color=logocolors.blue,
      label="EPCA",
      dpi=600
)

label_size = 8
tick_size = 8
xlabel!("State", fontsize=label_size)
ylabel!("Probability", fontsize=label_size)
title!("Belief Reconstructions", fontsize=14)

xlims!(0, 200)
ylims!(0.003, 0.008)

plot!(legend=:topright, legendfontsize=10, framestyle=:box)
figure2 = plot!(xtickfontsize=tick_size, ytickfontsize=tick_size)
# savefig("ExpFamilyPCA/scripts/reconstructions.png")
savefig("reconstructions.png")


epca_data = Dict(parse(Int, k) => v for (k, v) in JSON.parsefile("ExpFamilyPCA/scripts/jsons/epca_data.json"))
pca_data = Dict(parse(Int, k) => v for (k, v) in JSON.parsefile("ExpFamilyPCA/scripts/jsons/pca_data.json"))

kl_divs_poisson_epca = []
kl_divs_gaussian_epca = []

outdims = 1:6
for k in outdims
    push!(kl_divs_poisson_epca, mean(epca_data[k]))
    push!(kl_divs_gaussian_epca, mean(pca_data[k]))
end

plot(
    outdims, 
    kl_divs_poisson_epca, 
    label="EPCA", 
    color=logocolors.blue,
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
    color=logocolors.green,
    marker=:x,
    linestyle=:dash,
    lw=2,
    dpi=600

)
title!("KL Divergence Across Bases", fontsize=14)
xlabel!("Number of Bases", fontsize=label_size)
ylabel!("Average KL Divergence", fontsize=label_size)
figure1 = plot!(
    legendfontsize = 10,
    xtickfontsize = tick_size,
    ytickfontsize = tick_size,
    framestyle = :box,
    aspectration = :equal,
    markerstrokewidth = 2,  # Makes markers more prominent
    markeralpha = 0.7  # Adds transparency to markers
)
# savefig("ExpFamilyPCA/scripts/kl_divergence_plot.png")
savefig("kl_divergence_plot.png")


page_length_in_pixels = 11 * 72  # 11 inches at 72 DPI ~ 800 pixels
combined_plot = plot(figure1, figure2, layout = (1, 2),  size = (page_length_in_pixels, 400), left_margin = 2 * Plots.mm, bottom_margin = 2 * Plots.mm, dpi=800)
# savefig("ExpFamilyPCA/scripts/combo.png")
savefig("combo.png")