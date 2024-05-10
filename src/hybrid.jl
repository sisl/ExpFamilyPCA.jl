mutable struct ExplicitEPCA <: EPCA
    V
    Bregman  # Bregman divergence
    g::Function  # link function

    # hyperparameters
    μ
    ϵ
end