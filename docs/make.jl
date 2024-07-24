using Documenter
using ExpFamilyPCA

makedocs(
    sitename = "ExpFamilyPCA",
    checkdocs = :exports,
    format = Documenter.HTML(),
    modules = [ExpFamilyPCA],
    pages = [
        "ExpFamilyPCA.jl" => "index.md",
        "Math" => "math.md"
    ]
)