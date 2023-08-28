using Aqua
using BoxCox
using CairoMakie
using MixedModels
using MixedModels: dataset
using RDatasets: dataset as rdataset
using StatsModels
using Test

struct FakeTransformation <: BoxCox.PowerTransformation end
path(x) = joinpath(@__DIR__, "out", x)

@testset "Aqua" begin
    @static if VERSION >= v"1.9"
        Aqua.test_all(BoxCox; ambiguities=false, piracy=true)
    end
end

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

@testset "confint" begin
    y = [-0.174865, -0.312804, -1.06157, 1.20795, 0.573458, 0.0566415, 0.0481339, 1.98065,
         -0.196412, -0.464189]
    y2 = abs2.(y)
    X = ones(length(y), 1)
    # > y <- c(-0.174865, -0.312804, -1.06157, 1.20795, 0.573458, 0.0566415, 0.0481339, 1.98065,
    #      -0.196412, -0.464189)
    # > y2 <- y * y
    # > bc <- boxcox(y2 ~ 1, data=data.frame(y, y2), lambda=seq(-1, 1, 1e-5))
    # > bc$x[which.max(bc$y)]
    # [1] 0.06358
    # > ci <- bc$x[bc$y > max(bc$y) - 1/2 * qchisq(.95,1)]
    # > c(min(ci), max(ci))
    # [1] -0.22141  0.35783

    ci = [-0.22141, 0.35783]
    bc1 = fit(BoxCoxTransformation, y2)
    bc2 = fit(BoxCoxTransformation, X, y2)
    for bc in [bc1, bc2]
        @test only(params(bc)) ≈ 0.06358 atol = 1e-5
        @test all(isapprox.(confint(bc; fast=false), ci; atol=1e-4))
    end
    for bc in [bc1, bc2]
        @test all(isapprox.(confint(bc; fast=true), ci; atol=1e-2))
    end
end

@testset "plotting" begin
    vol = fit(BoxCoxTransformation, trees.Volume)
    qq = qqnorm(vol)
    save(path("qq.png"), qq)

    p = plot(vol)
    save(path("plot.png"), p)

    bcp = boxcoxplot(vol)
    save(path("boxcox.png"), bcp)

    volform = fit(BoxCoxTransformation,
                  @formula(Volume ~ 1 + log(Height) + log(Girth)),
                  trees)
    bcpf = Figure()
    ax = Axis(bcpf[1, 1])
    boxcoxplot!(ax, volform; conf_level=0.95,
                title="profile log likelihood",
                xlabel="parameter",
                ylabel="LL")
    save(path("boxcox_formula.png"), bcpf)

    @test_throws ArgumentError boxcoxplot(FakeTransformation())
    @test_throws ArgumentError boxcoxplot!(ax, FakeTransformation())
end

@testset "boxcox function" begin
    # log
    @test boxcox(0, 1) == boxcox(0)(1) == 0
    @test boxcox(1e-3, 1; atol=1e-2) == boxcox(1e-3; atol=1e-2)(1) == 0
    @test boxcox(1, 0) == -1
    @test boxcox(2, 0) == -1 / 2
end

@testset "show" begin
    bc = BoxCoxTransformation(; λ=1, y=[], X=nothing)

    output = """Box-Cox transformation

estimated λ: 1.0000
resultant transformation:

y (the identity)
"""
    @test sprint(show, bc) == output

    bc = BoxCoxTransformation(; λ=1e-3, y=[], X=nothing, atol=1e-2)
    output = """Box-Cox transformation

estimated λ: 0.0010
resultant transformation:

log y
"""

    @test sprint(show, bc) == output

    bc = BoxCoxTransformation(; λ=2, y=[], X=nothing, atol=1e-2)

    output = """Box-Cox transformation

estimated λ: 2.0000
resultant transformation:

 y^2.0 - 1
-----------
    2.0
"""
    @test sprint(show, bc) == output
end

@testset "mixed models" begin
    progress = false
    model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
                dataset(:sleepstudy); progress)
    bc = fit(BoxCoxTransformation, model; progress)
    @test only(params(bc)) ≈ -1 atol = 0.1
    ci = confint(bc; fast=false)
    ref_ci = [-1.73449, -0.413651]
    @test all(isapprox.(confint(bc; fast=true), ci; atol=1e-2))
    @test all(isapprox.(ci, ref_ci; atol=1e-6))

    @testset "mixed models + makie integration" begin
        bcpmm = boxcoxplot(bc; conf_level=0.95, title="sleep study should use speed")
        @test bcpmm isa Makie.FigureAxisPlot
        save(path("boxcox_mixedmodel.png"), bcpmm)
    end
end
