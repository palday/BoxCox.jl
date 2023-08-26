module BoxCoxMakieExt

using BoxCox
using Makie

using BoxCox: _loglikelihood_boxcox!,
              qr, chisqinvcdf, @compat,
              @setup_workload, @compile_workload

# XXX it would be great to have a 1-1 aspect ratio here,
# but this seems like something that should be done upstream
function Makie.convert_arguments(P::Type{<:Makie.QQNorm}, x::BoxCoxTransformation, args...;
                                 qqline=:fitrobust, kwargs...)
    return convert_arguments(P, x.(x.y), args...; qqline, kwargs...)
end

function Makie.convert_arguments(P::Type{<:Union{Makie.Scatter,Makie.Lines}},
                                 bc::BoxCoxTransformation, args...;
                                 λ=nothing, n_steps=21, kwargs...)
    if isnothing(λ)
        λ = range(-4 * abs(bc.λ), 4 * abs(bc.λ); length=n_steps)
    end
    sort!(collect(λ))

    @compat (; X, y) = bc
    ll = loglikelihood_boxcox(X, y, λ)
    return convert_arguments(P, λ, ll, args...; kwargs...)
end

function loglikelihood_boxcox(X::AbstractMatrix{<:Number}, y::Vector{<:Number}, λ)
    y_trans = similar(y)
    ll = similar(λ)
    Xqr = qr(X)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, Xqr, X, y, λ[i])
    end
    return ll
end

function loglikelihood_boxcox(::Nothing, y::Vector{<:Number}, λ)
    y_trans = similar(y)
    ll = similar(λ)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, y, λ[i])
    end
    return ll
end

function Makie.convert_arguments(P::Type{<:Makie.VLines}, bc::BoxCoxTransformation, args...;
                                 kwargs...)
    return convert_arguments(P, bc.λ, args...; kwargs...)
end

@recipe(BCPlot, boxcox) do scene
    s_theme = default_theme(scene, Scatter)
    l_theme = default_theme(scene, Lines)
    automatic = Makie.automatic
    scatline = Attributes(; color=l_theme.color,
                          colormap=l_theme.colormap,
                          # colorscale = l_theme.colorscale,
                          colorrange=get(l_theme.attributes, :colorrange, automatic),
                          linestyle=l_theme.linestyle,
                          linewidth=l_theme.linewidth,
                          markercolor=automatic,
                          markercolormap=theme(scene, :colormap),
                          markercolorrange=get(s_theme.attributes, :colorrange, automatic),
                          markersize=s_theme.markersize,
                          strokecolor=s_theme.strokecolor,
                          strokewidth=s_theme.strokewidth,
                          marker=s_theme.marker,
                          inspectable=theme(scene, :inspectable),
                          cycle=[:color])
    return merge(scatline, Attributes(; conf_level=0.95, n_steps=10, λ=nothing))
end

function Makie.plot!(p::BCPlot)
    # markercolor is the same as linecolor if left automatic
    # RGBColors -> union of all colortypes that `to_color` accepts + returns
    real_markercolor = Observable{Makie.RGBColors}()
    map!(real_markercolor, p.color, p.markercolor) do col, mcol
        if mcol === Makie.automatic
            return to_color(col)
        else
            return to_color(mcol)
        end
    end

    bc = p[1][]
    n_steps = p.n_steps[]
    λ = p.λ[]
    # TODO use splines
    scatterlines!(p, bc; λ, n_steps,
                  p.strokecolor, p.strokewidth, p.marker, p.markersize,
                  color=real_markercolor, p.linestyle, p.linewidth, p.colormap, # p.colorscale,
                  p.colorrange, p.inspectable)
    # seem to be hitting some buggy behavior in Makie
    # vlines!(p, bc; λ,
    #         p.color, linestyle=:dash, p.linewidth, p.colormap, # p.colorscale,
    #         p.colorrange, p.inspectable)
    return plot
end

BoxCox.boxcoxplot!(ax, bc::BoxCoxTransformation; kwargs...) = bcplot!(ax, bc; kwargs...)
BoxCox.boxcoxplot(bc::BoxCoxTransformation; kwargs...) = bcplot(bc; kwargs...)

function Makie.plot!(ax::Axis, P::Type{<:BCPlot}, allattrs::Makie.Attributes, bc)
    allattrs = merge(default_theme(P), allattrs)
    plot = Makie.plot!(ax.scene, P, allattrs, bc)

    if haskey(allattrs, :title)
        ax.title = allattrs.title[]
    end
    if haskey(allattrs, :xlabel)
        ax.xlabel = allattrs.xlabel[]
    else
        ax.xlabel = "λ"
    end
    if haskey(allattrs, :ylabel)
        ax.ylabel = allattrs.ylabel[]
    else
        ax.ylabel = "log likelihood"
    end
    # scatterlines!(ax, bc)
    # the ylim error doesn't happen if we do this here
    vlines!(ax, bc; bc.λ, linestyle=:dash)
    if haskey(allattrs, :conf_level)
        lltarget = loglikelihood(bc) - chisqinvcdf(1, allattrs.conf_level[]) / 2
        hlines!(ax, lltarget; linestyle=:dash)
    end

    return plot
end

Makie.plottype(::BoxCoxTransformation) = BCPlot

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
