using ExpFamilyPCA
using CompressedBeliefMDPs
using POMDPs, POMDPTools
using StableRNGs
using JSON
using Plots
using Statistics

const MAX_OUTDIMS = 6
const LOGO_COLORS = Colors.JULIA_LOGO_COLORS

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


function make_figure1()
    n_corridors = 2
    corridor_length = 20
    maze = CircularMaze(n_corridors, corridor_length)

    rng = StableRNG(100)
    policy = RandomPolicy(maze; rng=rng)
    sampler = PolicySampler(maze; policy=policy, rng=rng, n=200)

    raw_beliefs = sampler(maze)
    beliefs = make_numerical(raw_beliefs, maze)

    _, indim = size(beliefs)

    outdims = 1:MAX_OUTDIMS

    kl_divs_poisson_epca = []
    kl_divs_gaussian_epca = []

    for outdim in outdims
        @show outdim
        epca_kl = calc_kl(PoissonEPCA(indim, outdim), beliefs)
        pca_kl = calc_kl(GaussianEPCA(indim, outdim), beliefs)
        push!(kl_divs_poisson_epca, epca_kl)
        push!(kl_divs_gaussian_epca, pca_kl)
    end

    plot(
        outdims, 
        kl_divs_poisson_epca, 
        label="EPCA", 
        color=LOGO_COLORS.blue,
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
        color=LOGO_COLORS.green,
        marker=:x,
        linestyle=:dash,
        lw=2,
        dpi=600

    )
    label_size = 8
    tick_size = 8
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

    return figure1
end

function is_terminal_belief(belief)
    return all(x -> x == 0, belief[1:end-1]) && belief[end] == 1
end

function drop_terminal_beliefs(B, pomdp)
    B_numerical = make_numerical(B, pomdp)
    B_numerical = filter(row -> !is_terminal_belief(row), eachrow(B_numerical))  # exclude belief in terminal state
    B_numerical = reduce(hcat, B_numerical)'
    B_numerical = B_numerical[:, 1:end - 1]
    return B_numerical
end

function make_figure2(outdim=10)
    rng = StableRNG(123)
    pomdp = CircularMaze(2, 100)
    policy = RandomPolicy(pomdp; rng = rng)
    sampler = PolicySampler(pomdp; policy = policy, rng = rng, n = 40)
    beliefs = drop_terminal_beliefs(sampler(pomdp), pomdp)
    
    x = 1:length(states(pomdp)) - 1
    b1 = beliefs[1, :]
    n, indim = size(beliefs)

    pca = GaussianEPCA(indim, outdim)
    A1 = ExpFamilyPCA.fit!(pca, beliefs; verbose=true)
    recon1 = decompress(pca, A1)

    epca = PoissonEPCA(indim, outdim)
    A2 = ExpFamilyPCA.fit!(epca, beliefs; verbose=true)
    recon2 = decompress(epca, A2)

    plot(x, b1, 
     linestyle=:solid, 
     linewidth=1, 
     color=LOGO_COLORS.red,
     label="Original",
     dpi=600
    )

    plot!(x, recon1[1, :], 
        linestyle=:solid, 
        linewidth=2, 
        color=LOGO_COLORS.green,
        label="PCA",
        dpi=600
    )

    plot!(x, recon2[1, :], 
        linestyle=:dash, 
        linewidth=3, 
        color=LOGO_COLORS.blue,
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
    return figure2
end

# combine both figures
figure1 = make_figure1()
figure2 = make_figure2()
page_length_in_pixels = 11 * 72  # 11 inches at 72 DPI ~ 800 pixels
combined_plot = plot(figure1, figure2, layout = (1, 2),  size = (page_length_in_pixels, 400), left_margin = 2 * Plots.mm, bottom_margin = 2 * Plots.mm, dpi=800)
savefig("combo.png")




