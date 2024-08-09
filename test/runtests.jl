include("setup_tests.jl")

@testset ExtendedTestSet "Aqua" begin
    Aqua.test_all(BoxCox; ambiguities=false, piracy=true)
end

@testset ExtendedTestSet "Box-Cox" include("boxcox.jl")
