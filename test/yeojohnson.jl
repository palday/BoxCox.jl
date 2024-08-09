# taken from the Yeo-Johnson paper
plants = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0,  3.0, 9.3, 7.5, -6.0]
λref = 1.305
μ = 4.570
σ² = 29.876
lrt = 3.873
p = 0.0499

@testset "transformation and log-likelihood" begin
    yt0 = YeoJohnsonTransformation(; λ=1, X=nothing, y=plants)
    yt1 = YeoJohnsonTransformation(; λ=1.305, X=nothing, y=plants)
    @test yt0.(plants) ≈ plants
    @test isapprox(mean(yt1.(plants)),
                   μ; rtol=0.005)
    @test isapprox(var(yt1.(plants); corrected=false),
                   σ²; rtol=0.005)
    @test isapprox(2 * abs(loglikelihood(yt0) - loglikelihood(yt1)),
                   lrt; rtol=0.005)
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

# @testset "confint" begin
#     y = [-0.174865, -0.312804, -1.06157, 1.20795, 0.573458, 0.0566415, 0.0481339, 1.98065,
#          -0.196412, -0.464189]
#     y2 = abs2.(y)
#     X = ones(length(y), 1)
#     # > y <- c(-0.174865, -0.312804, -1.06157, 1.20795, 0.573458, 0.0566415, 0.0481339, 1.98065,
#     #      -0.196412, -0.464189)
#     # > y2 <- y * y
#     # > bc <- boxcox(y2 ~ 1, data=data.frame(y, y2), lambda=seq(-1, 1, 1e-5))
#     # > bc$x[which.max(bc$y)]
#     # [1] 0.06358
#     # > ci <- bc$x[bc$y > max(bc$y) - 1/2 * qchisq(.95,1)]
#     # > c(min(ci), max(ci))
#     # [1] -0.22141  0.35783

#     ci = [-0.22141, 0.35783]
#     bc1 = fit(BoxCoxTransformation, y2)
#     bc2 = fit(BoxCoxTransformation, X, y2)
#     for bc in [bc1, bc2]
#         @test only(params(bc)) ≈ 0.06358 atol = 1e-5
#         @test all(isapprox.(confint(bc; fast=false), ci; atol=1e-4))
#     end
#     for bc in [bc1, bc2]
#         @test all(isapprox.(confint(bc; fast=true), ci; atol=1e-2))
#     end
# end

# @testset "plotting" begin
#     vol = fit(BoxCoxTransformation, trees.Volume)
#     qq = qqnorm(vol)
#     @test qq isa Makie.FigureAxisPlot
#     save(path("qq.png"), qq)

#     qqfig = Figure(; title="QQNorm Mutating")
#     qqnorm!(Axis(qqfig[1, 1]; xlabel="Theoretical Quantiles", ylabel="Observed Quantiles"),
#             vol)
#     @test qqfig isa Makie.Figure
#     save(path("qqfig.png"), qqfig)

#     bcp = boxcoxplot(vol)
#     save(path("boxcox.png"), bcp)
#     @test bcp isa Makie.Figure

#     volform = fit(BoxCoxTransformation,
#                   @formula(Volume ~ 1 + log(Height) + log(Girth)),
#                   trees)
#     bcpf = Figure()
#     ax = Axis(bcpf[1, 1]; title="profile log likelihood")
#     boxcoxplot!(ax, volform; conf_level=0.95,
#                 xlabel="parameter",
#                 ylabel="LL")
#     save(path("boxcox_formula.png"), bcpf)

#     @test_throws ArgumentError boxcoxplot(FakeTransformation())
#     @test_throws ArgumentError boxcoxplot!(ax, FakeTransformation())
# end

# @testset "boxcox function" begin
#     # log
#     @test boxcox(0, 1) == boxcox(0)(1) == 0
#     @test boxcox(1e-3, 1; atol=1e-2) == boxcox(1e-3; atol=1e-2)(1) == 0
#     @test boxcox(1, 0) == -1
#     @test boxcox(2, 0) == -1 / 2
# end

# @testset "show" begin
#     bc = BoxCoxTransformation(; λ=1, y=[], X=nothing)

#     output = """Box-Cox transformation

# estimated λ: 1.0000
# resultant transformation:

# y (the identity)
# """
#     @test sprint(show, bc) == output

#     bc = BoxCoxTransformation(; λ=1e-3, y=[], X=nothing, atol=1e-2)
#     output = """Box-Cox transformation

# estimated λ: 0.0010
# resultant transformation:

# log y
# """

#     @test sprint(show, bc) == output

#     bc = BoxCoxTransformation(; λ=2, y=[], X=nothing, atol=1e-2)

#     output = """Box-Cox transformation

# estimated λ: 2.0000
# resultant transformation:

#  y^2.0 - 1
# -----------
#     2.0
# """
#     @test sprint(show, bc) == output
# end

# @testset "mixed models" begin
#     progress = false
#     model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
#                 dataset(:sleepstudy); progress)
#     bc = fit(BoxCoxTransformation, model; progress)
#     @test only(params(bc)) ≈ -1 atol = 0.1
#     ci = confint(bc; fast=false)
#     ref_ci = [-1.73449, -0.413651]
#     @test all(isapprox.(confint(bc; fast=true), ci; atol=1e-2))
#     @test all(isapprox.(ci, ref_ci; atol=1e-2))

#     @testset "mixed models + makie integration" begin
#         bcpmm = boxcoxplot(bc; conf_level=0.95)
#         @test bcpmm isa Makie.Figure
#         save(path("boxcox_mixedmodel.png"), bcpmm)
#     end
# end
