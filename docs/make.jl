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
        "Math" => [
            "Exponential Family PCA" => "math/epca.md",
            "Bregman Divergences" => "math/bregman.md",
            "EPCA Objective Derivations" => "math/derivations.md",
            "Gamma EPCA and the Itakura-Saito Distance" => "math/gamma.md",
        ],
        "Constructors" => [
            "Bernoulli" => "constructors/bernoulli.md",
            "Binomial" => "constructors/binomial.md",
            "Continuous Bernoulli" => "constructors/continuous_bernoulli.md",
            "Gamma" => "constructors/gamma.md",
            "Gaussian" => "constructors/gaussian.md",
            "Negative Binomial" => "constructors/negative_binomial.md",
            "Pareto" => "constructors/pareto.md",
            "Poisson" => "constructors/poisson.md",
            "Weibull" => "constructors/weibull.md",
        ],
        "API Documentation" => "api.md"
    ];
    plugins = [
        bib
    ]
)

deploydocs(
    repo = "github.com/FlyingWorkshop/ExpFamilyPCA.jl.git",
)

""" 
NEW

https://m3g.github.io/JuliaNotes.jl/stable/publish_docs/

julia> ] activate docs

julia> using ExpFamilyPCA; using LiveServer 

julia> servedocs()

OLD
To make the documentation run `julia --project make.jl` from the docs/ folder.
To view the the documentation locally, run `julia -e 'using LiveServer; serve(dir="docs/build")'` from the ExpFamilyPCA/ folder
"""