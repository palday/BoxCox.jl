# taken from the Yeo-Johnson paper
plants = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0,  3.0, 9.3, 7.5, -6.0]
λref = 1.305
μ = 4.570
σ² = 29.876
lrt = 3.873
p = 0.0499

@testset "transformation and log-likelihood" begin
    yt0 = YeoJohnsonTransformation(; λ=1, X=nothing, y=plants)
    yt1 = YeoJohnsonTransformation(; λ=λref, X=nothing, y=plants)
    @test yt0.(plants) ≈ plants
    @test isapprox(mean(yt1.(plants)),
                   μ; rtol=0.005)
    @test isapprox(var(yt1.(plants); corrected=false),
                   σ²; rtol=0.005)
    @test isapprox(2 * abs(loglikelihood(yt0) - loglikelihood(yt1)),
                   lrt; rtol=0.005)
    @test isapprox(pvalue(yt1), p; atol=0.005)
    @test yeojohnson(λref).(plants) ≈ yt1.(plants)
end

@testset "single vector" begin
    ytref = YeoJohnsonTransformation(; λ=1.305, X=nothing, y=plants)
    yt = fit(YeoJohnsonTransformation, plants)
    @test isapprox(only(params(yt)), only(params(ytref)); rtol=0.005)
    @test nobs(yt) == length(plants)
    @test nobs(empty!(yt)) == 0
    @test isempty(yt)
end

@testset "QR decomposition" begin
    ytref = YeoJohnsonTransformation(; λ=1.305, X=nothing, y=plants)
    plants_table = (; plants)
    yt = fit(YeoJohnsonTransformation, @formula(plants ~ 1), plants_table)
    @test isapprox(only(params(yt)), only(params(ytref)); rtol=0.005)
    @test loglikelihood(yt) ≈ loglikelihood(ytref) rtol=0.005
    @test nobs(yt) == length(plants)
    @test nobs(empty!(yt)) == 0
    @test isempty(yt)
end

@testset "confint: $name" for (name, X) in zip(["marginal", "conditional"], [nothing, ones(length(plants), 1)])
    yt1 = YeoJohnsonTransformation(; λ=λref, X, y=plants)
    ci = confint(yt1; fast=false)
    @test first(ci) < only(params(yt1)) < last(ci)

    fastci = confint(yt1; fast=true)
    @test first(ci) ≈ first(fastci)
    @test last(ci) ≈ last(fastci) rtol=0.05
end

@testset "plotting" begin
    yt1 = YeoJohnsonTransformation(; λ=λref, X=nothing, y=plants)
    qq = qqnorm(yt1)
    @test qq isa Makie.FigureAxisPlot
    save(path("qq-yeojohnson.png"), qq)

    qqfig = Figure(; title="QQNorm Mutating")
    qqnorm!(Axis(qqfig[1, 1]; xlabel="Theoretical Quantiles", ylabel="Observed Quantiles"),
            yt1)
    @test qqfig isa Makie.Figure
    save(path("qqfig-yeojohnson.png"), qqfig)

    p = boxcoxplot(yt1; conf_level=0.95)
    save(path("yeojohnson.png"), p)
    @test p isa Makie.Figure

    yt1 = YeoJohnsonTransformation(; λ=λref, X=ones(length(plants), 1), y=plants)
    p = boxcoxplot(yt1; conf_level=0.95)
    save(path("yeojohnson-matrix.png"), p)
    @test p isa Makie.Figure
end

@testset "show" begin
    yt = YeoJohnsonTransformation(; λ=1, y=[], X=nothing)
    output = """Yeo-Johnson transformation

estimated λ: 1.0000
resultant transformation:

y (the identity)
"""
    yt = YeoJohnsonTransformation(; λ=0.0001, y=[], X=nothing, atol=1e-2)
    output = """Yeo-Johnson transformation

estimated λ: 0.0001
resultant transformation:

For y ≥ 0,

log(y + 1)


For y < 0:

 -((-y + 1)^(2 - 0.0) - 1)
---------------------------
         (2 - 0.0)
"""

    @test sprint(show, yt) == output

    yt = YeoJohnsonTransformation(; λ=2.0001, y=[], X=nothing, atol=1e-2)
    output = """Yeo-Johnson transformation

estimated λ: 2.0001
resultant transformation:

For y ≥ 0,

 (y + 1)^2.0 - 1
-----------------
       2.0


For y < 0:

-log(-y + 1) for y < 0
"""

    @test sprint(show, yt) == output
    yt = YeoJohnsonTransformation(; λ=1.305, X=nothing, y=plants)

    output = """Yeo-Johnson transformation

estimated λ: 1.3050
p-value: 0.0488

resultant transformation:

For y ≥ 0,

 (y + 1)^1.3 - 1
-----------------
       1.3


For y < 0:

 -((-y + 1)^(2 - 1.3) - 1)
---------------------------
         (2 - 1.3)
"""
    @test sprint(show, yt) == output
end

@testset "mixed models" begin
    progress = false
    model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
                dataset(:sleepstudy); progress)
    yt = fit(YeoJohnsonTransformation, model; progress)

    # since the response values are all positive, this should essentially reduce
    # to the Box-Cox transformation and so we use those as reference values
    @test only(params(yt)) ≈ -1 atol = 0.1
    ci = confint(yt; fast=false)
    ref_ci = [-1.73449, -0.413651]
    @test all(isapprox.(confint(yt; fast=true), ci; atol=0.01))
    @test all(isapprox.(ci, ref_ci; rtol=0.05))

    @testset "mixed models + makie integration" begin
        p = boxcoxplot(yt; conf_level=0.95)
        @test p isa Makie.Figure
        save(path("yeojohnson_mixedmodel.png"), p)
    end
end
