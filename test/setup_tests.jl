using Aqua
using BoxCox
using CairoMakie
using MixedModels
using MixedModels: dataset
using RDatasets: dataset as rdataset
using StatsModels
using Test
using TestSetExtensions

struct FakeTransformation <: BoxCox.PowerTransformation end
path(x) = joinpath(@__DIR__, "out", x)
trees = rdataset("datasets", "trees")
