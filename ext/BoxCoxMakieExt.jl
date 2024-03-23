module BoxCoxMakieExt

using BoxCox
using Makie

using BoxCox: _loglikelihood_boxcox!,
              _loglikelihood_boxcox,
              qr, chisqinvcdf,
              @setup_workload, @compile_workload

# XXX it would be great to have a 1-1 aspect ratio here,
# but this seems like something that should be done upstream
function Makie.qqnorm!(ax::Axis, bc::BoxCoxTransformation, args...; kwargs...)
    return qqnorm!(ax, bc.(bc.y), args...; kwargs...)
end

function Makie.qqnorm(bc::BoxCoxTransformation, args...; kwargs...)
    return qqnorm(bc.(bc.y), args...; kwargs...)
end

function BoxCox._loglikelihood_boxcox(X::AbstractMatrix{<:Number}, y::Vector{<:Number},
                                      λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    Xqr = qr(X)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, Xqr, X, y, λ[i])
    end
    return ll
end

function BoxCox._loglikelihood_boxcox(::Nothing, y::Vector{<:Number},
                                      λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, y, λ[i])
    end
    return ll
end

function BoxCox.boxcoxplot(bc::BoxCoxTransformation; kwargs...)
    fig = Figure()
    boxcoxplot!(Axis(fig[1, 1]), bc; kwargs...)
    return fig
end

function BoxCox.boxcoxplot!(ax::Axis, bc::BoxCoxTransformation;
                            xlabel="λ",
                            ylabel="log likelihood",
                            n_steps=21,
                            λ=nothing,
                            conf_level=nothing,
                            attributes...)
    ax.xlabel = xlabel
    ax.ylabel = ylabel

    ci = nothing

    if !isnothing(conf_level)
        lltarget = loglikelihood(bc) - chisqinvcdf(1, conf_level) / 2
        hlines!(ax, lltarget; linestyle=:dash, color=:black)
        ci = confint(bc; level=conf_level)
        vlines!(ax, ci; linestyle=:dash, color=:black)
        text = "$(round(Int, 100 * conf_level))% CI"
        text!(ax, first(ci) + 0.05 * abs(first(ci)), lltarget; text)
    end

    if isnothing(λ)
        ci = @something(ci, confint(bc; fast=true))
        lower = first(ci) - 0.05 * abs(first(ci))
        upper = last(ci) + 0.05 * abs(last(ci))
        λ = range(lower, upper; length=n_steps)
    end
    sort!(collect(λ))

    (; X, y) = bc
    ll = _loglikelihood_boxcox(X, y, λ)

    scatterlines!(ax, λ, ll; attributes...)
    vlines!(ax, bc.λ; linestyle=:dash, color=:black)

    return plot
end

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    # draw from Normal(0,1)
    y = [-0.174865, -0.312804, -1.06157, 1.20795, 0.573458, 0.0566415, 0.0481339, 1.98065,
         -0.196412, -0.464189]
    y2 = abs2.(y)
    X = ones(length(y), 1)
    b1 = fit(BoxCoxTransformation, y2)
    b2 = fit(BoxCoxTransformation, X, y2)
    @compile_workload begin
        qqnorm(b1)
        boxcoxplot(b1)
        boxcoxplot(b2)
    end
end

end # module
