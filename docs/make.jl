using Documenter
using DocStringExtensions
using BoxCox

makedocs(; root=joinpath(dirname(pathof(BoxCox)), "..", "docs"),
         sitename="BoxCox",
         doctest=true,
         strict=true,
         pages=["index.md",
                "mixed-models.md",
                "api.md"])

deploydocs(; repo="github.com/palday/BoxCox.jl", push_preview=true, devbranch="main")
