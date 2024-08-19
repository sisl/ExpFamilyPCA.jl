### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 03562b9c-5de5-11ef-3032-3378ff5185dd
begin
	using Pkg
	Pkg.activate(".")
	using ExpFamilyPCA
	using CompressedBeliefMDPs

	using BenchmarkTools
	using Random
	using Plots
	using Statistics
	using Distances

	Random.seed!(1)
end


# ╔═╡ ce927bf1-e2f2-427c-88c8-2daedeaedbef
begin	
	n_corridors = 2
	corridor_length = 50
	maze = CircularMaze(n_corridors, corridor_length)
	sampler = PolicySampler(maze, n=200)
	raw_beliefs = sampler(maze)
	beliefs = make_numerical(raw_beliefs, maze)
end

# ╔═╡ 2ae1ec2b-f94a-4004-a829-f69341fb1d3c
outdim = 3

# ╔═╡ be21a6de-0a72-40a0-b520-b0155c8d75a0
@benchmark ExpFamilyPCA.fit!(
	PoissonEPCA(size(beliefs)[2], outdim),
	beliefs;
	maxiter=30,
)

# ╔═╡ ea131af3-49f4-4036-8fea-e4dd71b358fa
@benchmark ExpFamilyPCA.fit!(
	EPCA(
	    size(beliefs)[2],
	    outdim,
	    Distances.gkl_divergence,
	    exp,
	    Val((:Bregman, :g));
	    μ=1,
	    ϵ=eps()
	),
	beliefs;
	maxiter=30,
)

# ╔═╡ 2f943c3d-f16e-4ee8-aba8-3c8c2fc902c8


# ╔═╡ Cell order:
# ╠═03562b9c-5de5-11ef-3032-3378ff5185dd
# ╠═ce927bf1-e2f2-427c-88c8-2daedeaedbef
# ╠═2ae1ec2b-f94a-4004-a829-f69341fb1d3c
# ╠═be21a6de-0a72-40a0-b520-b0155c8d75a0
# ╠═ea131af3-49f4-4036-8fea-e4dd71b358fa
# ╠═2f943c3d-f16e-4ee8-aba8-3c8c2fc902c8
