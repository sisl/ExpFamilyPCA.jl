# ExpFamilyPCA.jl

[![Build Status](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FlyingWorkshop/ExpFamilyPCA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev-Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://flyingworkshop.github.io/ExpFamilyPCA.jl/dev/)

**ExpFamilyPCA.jl** is a Julia package for performing [Exponential Family Principal Component Analysis (EPCA)](https://papers.nips.cc/paper_files/paper/2001/hash/f410588e48dc83f2822a880a68f78923-Abstract.html), a dimensionality reduction technique tailored for data that follows an exponential family distribution. The package provides a flexible framework for applying EPCA to various data types, including Gaussian (normal), Bernoulli (binary), Poisson (count, probabilistic), Gamma (positive, continuous), and other distributions. 

## Features

- **Flexible EPCA Framework**: Supports multiple exponential family distributions, making it suitable for diverse data types.
- **Customizable Models**: Easily extendable to include new distributions and custom link functions.
- **Efficient Optimization**: Implements gradient-based optimization routines for efficient fitting and compression of high-dimensional data.
- **Data Compression and Reconstruction**: Offers utilities to compress data into a lower-dimensional space and reconstruct it back to its original form.

## Installation

To install the package, use the Julia package manager. In the Julia REPL, type:

```julia
using Pkg
Pkg.add("ExpFamilyPCA")
```

## Usage

### Importing the Package

To start using ExpFamilyPCA, first import the package:

```julia
using ExpFamilyPCA
```

### Basic Example

Below is a basic example of how to perform EPCA on normally-distributed data:

```julia
# Define the EPCA model parameters
indim = 100  # Input dimension (number of features)
outdim = 10  # Reduced output dimension (number of principal components)

# Create a Normal EPCA model instance
poisson_epca = PoissonEPCA(indim, outdim)

# Generate some random normally-distributed data
X = randn(1000, indim)  # 1000 observations, each with 100 features

# Fit the model to the data
X_compressed = fit!(normal_epca_model, X; maxiter=200, verbose=true)

# Compress out-of-distribution data
Y = randn(100, indim)  # 500 out-of-distribution observations, each with 100 features
Y_compressed = compress(poisson_epca, X; maxiter=100, verbose=true)

# Decompress the data to approximate the original data
X_reconstructed = decompress(poisson_epca, X_compressed)
```

### Working with Other Distributions

The package also supports other distributions such as Bernoulli, Poisson, and Gamma.


### List of Supported Models

- `NormalEPCA`: For Gaussian-distributed data. Equivalent to the usual PCA.
- `BernoulliEPCA`: For binary data (0 or 1).
- `PoissonEPCA`: For probability profiles and natural-valued data.
- `GammaEPCA`: For positive, continuous data. The EPCA objective is the Itakura-Saito distance.

## Documentation

For detailed documentation on each function and additional examples, please refer to the [documentation](https://github.com/username/ExpFamilyPCA.jl).

## Contributing

Contributions are welcome! If you want to contribute, please fork the repository, create a new branch, and submit a pull request. Before contributing, please make sure to update tests as appropriate.