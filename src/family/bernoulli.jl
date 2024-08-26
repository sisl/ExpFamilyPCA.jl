function BernoulliEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 0.5,
    ϵ = eps(),
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = nothing,
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
)
    # assumes χ = ℕ
    F(x) = x * log(x + ϵ) + (1 - x) * log1p(ϵ - x)
    g(θ) = exp(θ) / (1 + exp(θ))
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

