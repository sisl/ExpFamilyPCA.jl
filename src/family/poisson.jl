# TODO: add checking for μ to make sure in proper range

function PoissonEPCA(
    indim::Integer,
    outdim::Integer;
    μ = 1,
    ϵ = eps(),
    V_init::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    A_init_value::Union{Real, Nothing} = nothing,
    A_lower::Union{Real, Nothing} = nothing,
    A_upper::Union{Real, Nothing} = nothing,
    V_lower::Union{Real, Nothing} = nothing,
    V_upper::Union{Real, Nothing} = nothing
)
    # assumes χ = ℕ
    F(x) = x * log(x + ϵ) - x
    g(θ) = exp(θ)
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

