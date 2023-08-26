using Aqua
using BoxCox
using CairoMakie
using RDatasets: dataset as rdataset
using StatsModels
using Test

path(x) = joinpath(@__DIR__, "out", x)

@testset "Aqua" begin
    @static if VERSION >= v"1.9"
        Aqua.test_all(BoxCox; ambiguities=false, piracy=true)
    end
end

# using BoxCox, TestEnv; TestEnv.activate()
# using CairoMakie
# using RDatasets: dataset as rdataset
# using StatsModels
# using Test

trees = rdataset("datasets", "trees")

@testset "single vector" begin
    # > bc <- boxcox(Volume ~ 1, data = trees,
    #             lambda = seq(-0.25, 0.25, length = 10000), plotit=TRUE)
    # > print(bc$x[which.max(bc$y)], digits=16)
    # [1] -0.07478247824782477
    # > print(max(bc$y), digits=16)
    # [1] -32.79120399355727

    λref = -0.07478247824782477
    llref = -32.79120399355727

    vol = fit(BoxCoxTransformation, trees.Volume)
    volform = fit(BoxCoxTransformation, @formula(Volume ~ 1), trees)
    @test vol ≈ volform atol = 1e-6

    for bc in [vol, volform]
        @test only(params(bc)) ≈ λref rtol = 1e-3
        @test loglikelihood(bc) ≈ llref rtol = 1e-3
    end

    @test nobs(vol) == size(trees, 1)
    @test nobs(empty!(vol)) == 0
    @test isempty(vol)
end

@testset "QR decomposition" begin
    # > bc <- boxcox(Volume ~ log(Height) + log(Girth), data = trees,
    #             lambda = seq(-0.25, 0.25, length = 10000), plotit=FALSE)
    # > print(bc$x[which.max(bc$y)], digits=16)
    # [1] -0.06733173317331734
    # > print(max(bc$y), digits=16)
    # [1] 26.409734148606

    bcmass = fit(BoxCoxTransformation, @formula(Volume ~ log(Height) + log(Girth)), trees)
    @test only(params(bcmass)) ≈ -0.06733173317331734 rtol = 1e-3
    @test loglikelihood(bcmass) ≈ 26.409734148606 rtol = 1e-3
end

@testset "plotting" begin
    vol = fit(BoxCoxTransformation, trees.Volume)
    qq = qqnorm(vol)
    save(path("qq.png"), qq)

    p = plot(vol)
    save(path("plot.png"), p)

    bcp = boxcoxplot(vol)
    save(path("boxcox.png"), bcp)
end
