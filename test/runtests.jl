using Aqua
using LinearAlgebra
using BoxCox
using Test

@testset "Aqua" begin
    # it's not piracy for StatsAPI.r2(::MixedModel), it's privateering!
    Aqua.test_all(BoxCox; ambiguities=false, piracy=true)
end
