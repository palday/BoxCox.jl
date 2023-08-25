using Aqua
using LinearAlgebra
using BoxCox
using Test

@testset "Aqua" begin
    Aqua.test_all(BoxCox; ambiguities=false, piracy=true)
end
