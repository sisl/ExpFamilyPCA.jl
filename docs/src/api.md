# API Documentation

```@meta
CurrentModule = ExpFamilyPCA
```

# Contents

```@contents
Pages = ["api.md"]
```

# Index

```@index
Pages = ["api.md"]
```

# Functions

The core of the `ExpFamilyPCA.jl` API is the `EPCA` abstract type. All supported and custom EPCA specifications are subtypes of `EPCA` and include three methods in their `EPCA` interface: `fit!`, `compress` and `decompress`.

```@docs
EPCA
fit!
compress
decompress
```

# Miscellaneous 

```@docs
EPCACompressor
```
