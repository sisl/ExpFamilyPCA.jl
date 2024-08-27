"""

NOTE: The product A*V must contain only negtive entries.
"""
function GammaEPCA(
    indim::Integer,
    outdim::Integer;
    μ=1,
    ϵ=eps(),
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = -1,
    A_lower::Union{Real, Nothing} = -Inf,
    A_upper::Union{Real, Nothing} = -eps(),
    V_lower::Union{Real, Nothing} = eps(),
    V_upper::Union{Real, Nothing} = Inf
)
    # χ = ℝ++
    F(x) = -1 - log(x + ϵ)
    g(θ) = -1 / (θ + ϵ)
    epca = EPCA(
        indim,
        outdim,
        F,
        g,
        Val((:F, :g));
        μ = μ,
        ϵ = ϵ,
        V_init = V_init,
        A_init_value = A_init_value,
        A_lower = A_lower,
        A_upper = A_upper,
        V_lower = V_lower,
        V_upper = V_upper
    )
    return epca
end

# Alias
function ItakuraSaitoEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps(),
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = ones(outdim, indim),
    A_init_value::Union{Real, Nothing} = -1,
    A_lower::Union{Real, Nothing} = -Inf,
    A_upper::Union{Real, Nothing} = -eps(),
    V_lower::Union{Real, Nothing} = eps(),
    V_upper::Union{Real, Nothing} = Inf
)
    epca = GammaEPCA(
        indim,
        outdim;
        μ = μ,
        ϵ = ϵ,
        V_init = V_init,
        A_init_value = A_init_value,
        A_lower = A_lower,
        A_upper = A_upper,
        V_lower = V_lower,
        V_upper = V_upper
    )
    return epca
end