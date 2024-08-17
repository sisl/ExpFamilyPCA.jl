### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 18fe4b4a-5c88-11ef-18b1-31b764bf2b80
begin
	import Pkg
	Pkg.activate(".")
	using ExpFamilyPCA

	# 3Ps
	using BenchmarkTools
	using Plots

	# POMDPs.jl
	using CompressedBeliefMDPs
	using POMDPTools
	using POMDPs
end

# ╔═╡ 04e5320e-a9ed-4b3f-b5ba-a51464d572b6
begin
	struct EPCACompressor <: Compressor
		epca::EPCA
	end
	
	function (c::EPCACompressor)(beliefs)
		compressed_beliefs = compress(c.epca, beliefs)
		return compressed_beliefs
	end
	
	function CompressedBeliefMDPs.fit!(c::EPCACompressor, beliefs)
		ExpFamilyPCA.fit!(c.epca, beliefs)
	end
end

# ╔═╡ 25f1a8a3-608f-4aa1-801d-ee747cef4fbc
begin
	n_corridors = 2
	corridor_length = 20
	maze = CircularMaze(n_corridors, corridor_length)
end

# ╔═╡ 799b1c4f-4bc7-4a46-97d6-01737f7e9117
@show Pkg.status("POMDPTools")

# ╔═╡ 5074f669-4345-4e42-aecb-9f70f19bc65d
begin
	indim = (n_corridors * corridor_length) + 1
	outdim = 10  # TODO: iterate across multiple values
	epca = PoissonEPCA(indim, outdim)
	compressor = EPCACompressor(epca)
	updater = DiscreteUpdater(maze)
	sampler = BeliefExpansionSampler(maze)
	solver = CompressedBeliefSolver(
	    maze;
	    compressor=compressor,
	    sampler=sampler,
	    updater=updater,
	    verbose=true, 
	    max_iterations=100, 
	    n_generative_samples=50, 
	    k=2
	)
end

# ╔═╡ Cell order:
# ╠═18fe4b4a-5c88-11ef-18b1-31b764bf2b80
# ╠═04e5320e-a9ed-4b3f-b5ba-a51464d572b6
# ╠═25f1a8a3-608f-4aa1-801d-ee747cef4fbc
# ╠═799b1c4f-4bc7-4a46-97d6-01737f7e9117
# ╠═5074f669-4345-4e42-aecb-9f70f19bc65d
