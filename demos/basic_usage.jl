using Revise

using ExpFamilyPCA


n_samples = 5
n_dims = 100
X = rand(0:1, n_samples, n_dims)  # generate random binary data

n_components = 2
epca = BernoulliPCA(n_components, n_dims)
fit!(epca, X; verbose=true, maxiter=5)

X̃ = compress(epca, X)
X_recon = decompress(epca, X̃)