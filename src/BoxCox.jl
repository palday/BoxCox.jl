module BoxCox

using Compat
using DocStringExtensions
using LinearAlgebra
# we use NLopt because that's what MixedModels uses and this was developed
# with a particular application of MixedModels.jl in mind
using NLopt
using PrecompileTools
using Printf
using Statistics
using StatsAPI
using StatsBase
using StatsFuns

using StatsAPI: params

export BoxCoxTransformation,
       boxcox,
       confint,
       boxcoxplot,
       boxcoxplot!,
       fit,
       loglikelihood,
       nobs,
       params

"""
    PowerTransformation

Abstract type representing [power transformations](https://en.wikipedia.org/wiki/Power_transform)
such as the Box-Cox transformation.
"""
abstract type PowerTransformation end
# struct YeoJohnsonTransformation <: PowerTransformation end
# struct BickelDoksumTransformation <: PowerTransformation end

"""
    struct BoxCoxTransformation <: PowerTransformation

# Fields

$(FIELDS)

!!! note
    All fields are considered internal and implementation details and may change at any time without
    being considered breaking.

# Tips

- To extract the λ parameter, use `params`.
- The transformation is callable, meaning that you can do
```@example
bc = fit(BoxCoxTransformation, y)
y_transformed = bc.(y)
```
- You can reduce the size of a BoxCoxTransformation in memory by using `empty!`, but certain diagnostics
  (e.g. plotting and computation of the loglikelihood will no longer be available).

See also [`boxcoxplot`](@ref), [`params`](@ref), [`boxcox`](@ref).
"""
Base.@kwdef struct BoxCoxTransformation{T} <: PowerTransformation
    "The transformation parameter"
    λ::Float64
    "The original response, normalized by its geometric mean"
    y::Vector{Float64} # observed response normalized by its geometric mean
    "A model matrix for the conditional distribution or `Nothing` for the unconditional distribution "
    X::T
    "Tolerance for comparing λ to zero. Default is 1e-8"
    atol::Float64 = 1e-8 # isapprox tolerance to round towards zero or one
end

function BoxCoxTransformation(λ::Number, y::Vector, X::T, atol::Number) where {T}
    return BoxCoxTransformation{T}(; λ, y, X, atol)
end

"""
    Base.isapprox(x::BoxCoxTransformation, y::BoxCoxTransformation; kwargs...)

Compare the λ parameter of `x` and `y` for approximate equality.

`kwargs` are passed on to `isapprox` for the parameters.

!!! note
    Other internal structures of `BoxCoxTransformation` are not compared.
"""
function Base.isapprox(x::BoxCoxTransformation, y::BoxCoxTransformation; kwargs...)
    return isapprox(x.λ, y.λ; kwargs...)
end

"""
    boxcox(λ; atol=0)
    boxcox(λ, x; atol=0)

Compute the Box-Cox transformation of x for the parameter value λ.

`atol` controls the absolute tolerance for treating λ as zero.

The one argument variant curries and creates a one-argument function of `x` for the given λ.

See also [BoxCoxTransformation](@ref).

# References

Box, George E. P.; Cox, D. R. (1964). "An analysis of transformations". _Journal of the Royal Statistical Society_, Series B. 26 (2): 211--252.
"""
boxcox(λ; kwargs...) = x -> boxcox(λ, x; kwargs...)
function boxcox(λ, x; atol=0)
    if isapprox(λ, 0; atol)
        logx = log(x)
        λlogx = λ * logx
        return logx * (1 + (λlogx) / 2 * (1 + (λlogx) / 3 * (1 + (λlogx) / 4)))
    end

    return (x^λ - 1) / λ
end

"""
    (t::BoxCoxTransformation)(x::Number)

Apply the estimated BoxCox transformation `t` to the number `x`.

See also [`BoxCox`](@ref).
"""
function (t::BoxCoxTransformation)(x::Number)
    return boxcox(t.λ, x)
end

"""
    boxcoxplot(bc::BoxCoxTransformation; kwargs...)
    boxcoxplot!(axis, bc::BoxCoxTransformation; λ=nothing, n_steps=21)

Create a diagnostic plot for the Box-Cox transformation.

If λ is `nothing`, the range of possible values for the λ paramter is automatically determined,
with a total of `n_steps`. If `λ` is a vector of numbers, then the λ parameter is evaluated at
each element of that vector.

!!! note
    You must load an appropriate Makie backend (e.g., CairoMakie or GLMakie) to actually render a plot.

!!! note
    A meaningful plot is only possible when `bc` has not been `empty!`'ed.

!!! compat "Julia 1.6"
    The plotting functionality is defined unconditionally.

!!! compat "Julia 1.9"
    The plotting functionality interface is defined as a package extension and only loaded when Makie is available.
"""
function boxcoxplot!(::Any, ::PowerTransformation; kwargs...)
    # specialize slightly so that they can't just throw Any and get this message
    throw(ArgumentError("Have you loaded an appropriate Makie backend?"))
end

"$(@doc boxcoxplot!)"
function boxcoxplot(::PowerTransformation; kwargs...)
    throw(ArgumentError("Have you loaded an appropriate Makie backend?"))
end

"""
    empty!(bt::BoxCoxTransformation)

Empty internal storage of `bt`.

For transformations fit to a large amount of data, this can reduce the size in memory.
However, it means that [`loglikelihood`](@ref), [`boxcoxplot`](@ref) and other functionality
dependent on having access to the original data will no longer work.

After emptying, `bt` can still be used to transform **new** data.
"""
function Base.empty!(bt::BoxCoxTransformation{T}) where {T}
    empty!(bt.y)
    # is there a way to make this work for matrices, mixed models, etc.?
    hasmethod(empty!, (T,)) && empty!(bt.X)
    return bt
end

Base.isempty(bt::BoxCoxTransformation) = any(isempty, [bt.y, something(bt.X, [])])

# TODO: Do more optimization error checking

"""
    StatsAPI.fit(::Type{BoxCoxTransformation}, y::AbstractVector{<:Number}; atol=1e-8,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                maxiter=-1)
    StatsAPI.fit(::Type{BoxCoxTransformation}, X::AbstractMatrix{<:Number},
                 y::AbstractVector{<:Number}; atol=1e-8,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                 maxiter=-1)
    StatsAPI.fit(::Type{BoxCoxTransformation}, formula::FormulaTerm, data;
                 atol=1e-8,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                 maxiter=-1)
    StatsAPI.fit(::Type{BoxCoxTransformation}, model::LinearMixedModel;
                 atol=1e-8, progress=true,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                 maxiter=-1)




Find the optimal λ value for a Box-Cox transformation of the data.

When no `X` is provided, `y` is treated as an unconditional distribution.

When `X` is provided, `y` is treated as distribution conditional on the linear predictor defined by `X`.
At each iteration step, a simple linear regression is fit to the transformed `y` with `X` as the model matrix.

If a `FormulaTerm` is provided, then `X` is constructed using that specification and `data`.

If a `LinearMixedModel` is provided, then `X` and `y` are extracted from the model object.

!!! note
    The formula interface is only available if StatsModels.jl is loaded either directly or via another package
    such GLM.jl or MixedModels.jl.

!!! compat "Julia 1.6"
    - The formula interface is defined unconditionally, but `@formula` is not loaded.
    - The MixedModels interface is defined unconditionally.

!!! compat "Julia 1.9"
    - The formula interface is defined as a package extension.
    - The MixedModels interface is defined as a package extension.

`atol` controls the absolute tolerance for treating `λ` as zero.

The `opt_` keyword arguments are tolerances passed onto NLopt.

`maxiter` specifies the maximum number of iterations to use in optimization; negative values place no restriciton.

`algorithm` is a valid NLopt algorithm to use in optimization.

`progress` enables progress bars for intermediate model fits during the optimization process.
"""
function StatsAPI.fit(::Type{BoxCoxTransformation}, y::AbstractVector{<:Number}; atol=1e-8,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1)
    any(<=(0), y) && throw(ArgumentError("all y values must be greater than zero"))
    y = float.(y)  # we modify, so let's make a copy!
    y ./= geomean(y)
    opt = NLopt.Opt(algorithm, 1)
    NLopt.xtol_abs!(opt, opt_atol) # relative criterion on parameter values
    NLopt.xtol_rel!(opt, opt_rtol) # relative criterion on parameter values
    NLopt.maxeval!(opt, maxiter)
    local y_trans = similar(y)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = _loglikelihood_boxcox!(y_trans, y, only(λvec))
        return val
    end
    opt.max_objective = obj
    (ll, λ, retval) = optimize(opt, [0.0])

    return BoxCoxTransformation(; λ=only(λ), y, X=nothing, atol)
end

function StatsAPI.fit(::Type{BoxCoxTransformation}, X::AbstractMatrix{<:Number},
                      y::AbstractVector{<:Number}; atol=1e-8,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1)
    any(<=(0), y) && throw(ArgumentError("all y values must be greater than zero"))
    y = float.(y) # we modify, so let's make a copy!
    y ./= geomean(y)
    X = convert(Matrix{Float64}, X)
    Xqr = qr(X)

    opt = NLopt.Opt(algorithm, 1)
    NLopt.xtol_abs!(opt, opt_atol) # relative criterion on parameter values
    NLopt.xtol_rel!(opt, opt_rtol) # relative criterion on parameter values
    NLopt.maxeval!(opt, maxiter)
    local y_trans = similar(y)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = _loglikelihood_boxcox!(y_trans, Xqr, X, y, only(λvec))
        return val
    end
    opt.max_objective = obj
    (ll, λ, retval) = optimize(opt, [1.0])
    return BoxCoxTransformation(; λ=only(λ), y, X, atol)
end

"""
    _boxcox!(y_trans, y, λ; kwargs...)

Internal method to compute `boxcox` at each element of `y` and store the result in `y_trans`.
"""
function _boxcox!(y_trans, y, λ; kwargs...)
    for i in eachindex(y, y_trans; kwargs...)
        y_trans[i] = boxcox(λ, y[i])
    end
    return y_trans
end

# pull this out so that we can use it in optimization
function _loglikelihood_boxcox!(y_trans::Vector{<:Number}, Xqr::Factorization,
                                X::Matrix{<:Number}, y::Vector{<:Number}, λ::Number;
                                kwargs...)
    _boxcox!(y_trans, y, λ; kwargs...)
    y_trans -= X * (Xqr \ y_trans)
    return _loglikelihood_boxcox(y_trans)
end

function _loglikelihood_boxcox!(y_trans::Vector{<:Number}, y::Vector{<:Number}, λ::Number;
                                kwargs...)
    _boxcox!(y_trans, y, λ; kwargs...)
    y_trans .-= mean(y_trans)
    return _loglikelihood_boxcox(y_trans)
end

function _loglikelihood_boxcox(y_trans::Vector{<:Number})
    return -0.5 * length(y_trans) * log(sum(abs2, y_trans))
end

function _loglikelihood_boxcox(λ::Number, X::AbstractMatrix{<:Number}, y::Vector{<:Number};
                               kwargs...)
    return _loglikelihood_boxcox!(similar(y), qr(X), X, y, λ)
end

function _loglikelihood_boxcox(λ::Number, ::Nothing, y::Vector{<:Number}; kwargs...)
    return _loglikelihood_boxcox!(similar(y), y, λ)
end

StatsAPI.nobs(bc::BoxCoxTransformation) = length(bc.y)

StatsAPI.params(bc::BoxCoxTransformation) = [bc.λ]

# function _pvalue(bc::BoxCoxTransformation)
#     llhat = loglikelihood(bc)
#     ll0 = _loglikelihood_boxcox(0, bc.X, bc.y)
#     return chisqcdf(1, 2 * (llhat - ll0))
# end

"""
    StatsAPI.confint(bc::BoxCoxTransformation; level::Real=0.95, fast::Bool=nobs(bc) > 10_000)

Compute confidence intervals for λ, with confidence level level (by default 95%).

If `fast`, then a symmetric confidence interval around ̂λ is assumed and the upper bound
is computed using the difference between the lower bound and λ. Symmetry is generally a
safe assumption for approximate values and halves computation time.

If not `fast`, then the lower and upper bounds are computed separately.
"""
function StatsAPI.confint(bc::BoxCoxTransformation; level::Real=0.95,
                          fast::Bool=nobs(bc) > 10_000)
    # ll0 = _loglikelihood_boxcox(0, bc.X, bc.y)

    lltarget = loglikelihood(bc) - chisqinvcdf(1, level) / 2
    opt = NLopt.Opt(:LN_BOBYQA, 1)
    Xqr = isnothing(bc.X) ? nothing : qr(bc.X)
    y_trans = similar(bc.y)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        llhat = if isnothing(bc.X)
            _loglikelihood_boxcox!(y_trans, bc.y, only(λvec))
        else
            _loglikelihood_boxcox!(y_trans, Xqr, bc.X, bc.y, only(λvec))
        end
        # want this to be zero
        val = abs(llhat - lltarget)
        return val
    end
    opt.min_objective = obj

    NLopt.upper_bounds!(opt, bc.λ)
    (ll, λvec, retval) = optimize(opt, [bc.λ - 1])
    lower = only(λvec)

    if fast
        upper = (bc.λ - lower) + bc.λ
    else
        NLopt.lower_bounds!(opt, bc.λ)
        NLopt.upper_bounds!(opt, Inf)
        (ll, λvec, retval) = optimize(opt, [bc.λ + 1])
        upper = only(λvec)
    end
    return [lower, upper]
end

function StatsAPI.loglikelihood(t::BoxCoxTransformation)
    return _loglikelihood_boxcox(t.λ, t.X, t.y; t.atol)
end

function Base.show(io::IO, t::BoxCoxTransformation)
    println(io, "Box-Cox transformation")
    @printf io "\nestimated λ: %.4f" t.λ
    # println(io, "\np-value: ", StatsBase.PValue(_pvalue(t)))
    println(io, "\nresultant transformation:\n")

    if isapprox(t.λ, 1; t.atol)
        println(io, "y (the identity)")
        return nothing
    end

    λ = @sprintf "%.1f" t.λ

    if isapprox(0, t.λ; t.atol)
        println(io, "log y")
        return nothing
    end

    numerator = "y^$(λ) - 1"
    denominator = λ
    width = maximum(length, [numerator, denominator]) + 2
    println(io, lpad(numerator, (width - length(numerator)) ÷ 2 + length(numerator)))
    println(io, "-"^width)
    println(io, lpad(denominator, (width - length(denominator)) ÷ 2 + length(denominator)))

    return nothing
end

if !isdefined(Base, :get_extension)
    include("../ext/BoxCoxMakieExt.jl")
    include("../ext/BoxCoxMixedModelsExt.jl")
    include("../ext/BoxCoxStatsModelsExt.jl")
end

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    # draw from Normal(0,1)
    y = [-0.174865, -0.312804, -1.06157, 1.20795, 0.573458, 0.0566415, 0.0481339, 1.98065,
         -0.196412, -0.464189]
    y2 = abs2.(y)
    X = ones(length(y), 1)

    @compile_workload begin
        fit(BoxCoxTransformation, y2)
        fit(BoxCoxTransformation, X, y2)
    end
end

end # module BoxCox
