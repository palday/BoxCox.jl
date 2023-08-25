using Documenter
using BoxCox

makedocs(; root=joinpath(dirname(pathof(BoxCox)), "..", "docs"),
         sitename="BoxCox",
         doctest=true,
         pages=["index.md", "api.md"])

deploydocs(; repo="github.com/palday/BoxCox.jl.git", push_preview=true)
