using Documenter
using ExpFamilyPCA
using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

makedocs(
    sitename = "ExpFamilyPCA",
    checkdocs = :exports,
    format = Documenter.HTML(),
    modules = [ExpFamilyPCA],
    pages = [
        "ExpFamilyPCA.jl" => "index.md",
        "Math" => "math.md",
        "Bregman Divergences" => "bregman.md",
        "API Documentation" => "api.md"
    ];
    plugins = [
        bib
    ]
)

""" 
NEW

https://m3g.github.io/JuliaNotes.jl/stable/publish_docs/

julia> ] activate docs

julia> using LiveServer 

julia> servedocs()

OLD
To make the documentation run `julia --project make.jl` from the docs/ folder.
To view the the documentation locally, run `julia -e 'using LiveServer; serve(dir="docs/build")'` from the ExpFamilyPCA/ folder
"""