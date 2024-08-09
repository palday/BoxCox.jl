module BoxCox

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

using StatsBase: PValue
# XXX I have no idea why this is necessary, but otherwise isdefined(BoxCox, :params) returns false
using StatsAPI: params, pvalue

export confint,
       fit,
       loglikelihood,
       nobs,
       params,
       pvalue

export boxcoxplot, boxcoxplot!

"""
    PowerTransformation

Abstract type representing [power transformations](https://en.wikipedia.org/wiki/Power_transform)
such as the Box-Cox transformation.
"""
abstract type PowerTransformation end
# struct BickelDoksumTransformation <: PowerTransformation end

include("BoxCoxTransformation.jl")
export BoxCoxTransformation, boxcox

include("YeoJohnsonTransformation.jl")
export YeoJohnsonTransformation, yeojohnson


#####
##### Base methods
#####

"""
    Base.isapprox(x::PowerTransformation, y::PowerTransformation; kwargs...)

Compare the λ parameter of `x` and `y` for approximate equality.

`kwargs` are passed on to `isapprox` for the parameters.

!!! note
    Other internal structures of `PowerTransformation` are not compared.
"""
function Base.isapprox(x::T, y::S; kwargs...) where {T <: PowerTransformation, S <: PowerTransformation}
    # XXX why not do this through the type signature? Well, we want to allow e.g.
    # BoxCoxTransformation{Nothing} and BoxCoxTransformation{Float32} to be compared
    typejoin(T, S) === PowerTransformation &&
        throw(ArgumentError("x and y must be the same subtype of PowerTransformation"))

    return all(isapprox.(params(x), params(y); kwargs...))
end

"""
    empty!(x::PowerTransformation)

Empty internal storage of `x`.

For transformations fit to a large amount of data, this can reduce the size in memory.
However, it means that [`loglikelihood`](@ref) as well as plotting and other functionality
dependent on having access to the original data will no longer work.

After emptying, `x` can still be used to transform **new** data.
"""
function Base.empty!(x::PowerTransformation)
    empty!(x.y)
    # is there a way to make this work for matrices, mixed models, etc.?
    hasmethod(empty!, (typeof(modelmatrix(x)),)) && empty!(modelmatrix(x))
    return x
end

Base.isempty(x::PowerTransformation) = any(isempty, [x.y, something(modelmatrix(x), [])])

#####
##### StatsAPI methods
#####


"""
    StatsAPI.confint(x::PowerTransformation; level::Real=0.95, fast::Bool=nobs(bc) > 10_000)

Compute confidence intervals for λ, with confidence level level (by default 95%).

If `fast`, then a symmetric confidence interval around ̂λ is assumed and the upper bound
is computed using the difference between the lower bound and λ. Symmetry is generally a
safe assumption for approximate values and halves computation time.

If not `fast`, then the lower and upper bounds are computed separately.
"""
function StatsAPI.confint(x::T; level::Real=0.95,
                          fast::Bool=nobs(x) > 10_000) where {T <: PowerTransformation}
    lltarget = loglikelihood(x) - chisqinvcdf(1, level) / 2
    opt = NLopt.Opt(:LN_BOBYQA, 1)
    X = modelmatrix(x)
    Xqr = isnothing(X) ? nothing : qr(modelmatrix(x))
    y_trans = similar(response(x))
    ll! = _llfunc!(T)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        llhat = if isnothing(X)
            ll!(y_trans, response(x), only(λvec))
        else
            ll!(y_trans, Xqr, X, response(x), only(λvec))
        end
        # want this to be zero
        val = abs(llhat - lltarget)
        return val
    end
    opt.min_objective = obj

    λ = only(params(x))
    NLopt.upper_bounds!(opt, λ)
    (ll, λvec, retval) = optimize(opt, [λ - 1])
    lower = only(λvec)

    if fast
        upper = (λ - lower) + λ
    else
        NLopt.lower_bounds!(opt, λ)
        NLopt.upper_bounds!(opt, Inf)
        (ll, λvec, retval) = optimize(opt, [λ + 1])
        upper = only(λvec)
    end
    return [lower, upper]
end

# TODO: Do more optimization error checking

"""
    StatsAPI.fit(::Type{<:PowerTransformation}, y::AbstractVector{<:Number}; atol=1e-8,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                maxiter=-1)
    StatsAPI.fit(::Type{<:PowerTransformation}, X::AbstractMatrix{<:Number},
                 y::AbstractVector{<:Number}; atol=1e-8,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                 maxiter=-1)
    StatsAPI.fit(::Type{<:PowerTransformation}, formula::FormulaTerm, data;
                 atol=1e-8,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                 maxiter=-1)
    StatsAPI.fit(::Type{<:PowerTransformation}, model::LinearMixedModel;
                 atol=1e-8, progress=true,
                 algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                 maxiter=-1)

Find the optimal λ value for a power transformation of the data.

When no `X` is provided, `y` is treated as an unconditional distribution.

When `X` is provided, `y` is treated as distribution conditional on the linear predictor defined by `X`.
At each iteration step, a simple linear regression is fit to the transformed `y` with `X` as the model matrix.

If a `FormulaTerm` is provided, then `X` is constructed using that specification and `data`.

If a `LinearMixedModel` is provided, then `X` and `y` are extracted from the model object.

!!! note
    The formula interface is only available if StatsModels.jl is loaded either directly or via another package
    such GLM.jl or MixedModels.jl.

!!! note
    - The formula interface is defined as a package extension.
    - The MixedModels interface is defined as a package extension.

`atol` controls the absolute tolerance for treating `λ` as zero.

The `opt_` keyword arguments are tolerances passed onto NLopt.

`maxiter` specifies the maximum number of iterations to use in optimization; negative values place no restrictions.

`algorithm` is a valid NLopt algorithm to use in optimization.

`progress` enables progress bars for intermediate model fits during the optimization process.
"""
function StatsAPI.fit(T::Type{<:PowerTransformation}, y::AbstractVector{<:Number}; atol=1e-8,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1)
    _input_check(T)(y)
    # we modify, so let's make a copy!
    y =  (y .- _centering(T)(y)) ./ _scaling(T)(y)
    opt = NLopt.Opt(algorithm, 1)
    NLopt.xtol_abs!(opt, opt_atol) # relative criterion on parameter values
    NLopt.xtol_rel!(opt, opt_rtol) # relative criterion on parameter values
    NLopt.maxeval!(opt, maxiter)
    local y_trans = similar(y)
    ll! = _llfunc!(T)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = ll!(y_trans, y, only(λvec))
        return val
    end
    opt.max_objective = obj
    (ll, λ, retval) = optimize(opt, [0.0])

    return T(; λ=only(λ), y, X=nothing, atol)
end

function StatsAPI.fit(T::Type{<:PowerTransformation}, X::AbstractMatrix{<:Number},
                      y::AbstractVector{<:Number}; atol=1e-8,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1)
    _input_check(T)(y)
    # we modify, so let's make a copy!
    y =  (y .- _centering(T)(y)) ./ _scaling(T)(y)
    X = convert(Matrix{Float64}, X)
    Xqr = qr(X)

    opt = NLopt.Opt(algorithm, 1)
    NLopt.xtol_abs!(opt, opt_atol) # relative criterion on parameter values
    NLopt.xtol_rel!(opt, opt_rtol) # relative criterion on parameter values
    NLopt.maxeval!(opt, maxiter)
    local y_trans = similar(y)
    ll! = _llfunc!(T)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = ll!(y_trans, Xqr, X, y, only(λvec))
        return val
    end
    opt.max_objective = obj
    (ll, λ, retval) = optimize(opt, [1.0])
    return T(; λ=only(λ), y, X, atol)
end

function StatsAPI.loglikelihood(t::T) where {T <: PowerTransformation}
    return _llfunc(T)(t.λ, t.X, t.y; t.atol)
end

StatsAPI.modelmatrix(x::PowerTransformation) = x.X
StatsAPI.nobs(x::PowerTransformation) = length(response(x))

"""
    StatsAPI.params(x::PowerTransformation)

Return a vector of all parameters, i.e. `[λ]`.
"""
StatsAPI.params(x::PowerTransformation) = [x.λ]
StatsAPI.pvalue(x::PowerTransformation) = 1 - chisqcdf(1, lrt0(x))
StatsAPI.response(x::PowerTransformation) = x.y

lrt0(x::PowerTransformation) = 2 * abs(loglikelihood(x) - loglikelihood(_identity(x)))

#####
##### Precompilation
#####


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
