using Documenter
using ExpFamilyPCA
using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

makedocs(
    sitename = "ExpFamilyPCA",
    # checkdocs = :exports,
    format = Documenter.HTML(),
    # modules = [ExpFamilyPCA],
    pages = [
        "ExpFamilyPCA.jl" => "index.md",
        "Math" => "math.md"
    ];
    plugins = [
        bib
    ]
)

"""
To make the documentation run `julia --project make.jl` from the docs/ folder.
To view the the documentation locally, run `julia -e 'using LiveServer; serve(dir="docs/build")'`
"""