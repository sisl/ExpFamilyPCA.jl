using Documenter
using ExpFamilyPCA

makedocs(
    sitename = "ExpFamilyPCA",
    # checkdocs = :exports,
    format = Documenter.HTML(),
    # modules = [ExpFamilyPCA],
    pages = [
        "ExpFamilyPCA.jl" => "index.md",
        "Math" => "math.md"
    ]
)

"""
To make the documentation run `julia --project make.jl` from the docs/ folder.
"""