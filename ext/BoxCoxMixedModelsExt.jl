module BoxCoxMixedModelsExt

using BoxCox
using MixedModels

using BoxCox: StatsAPI, NLopt
using BoxCox: _boxcox!, _yeojohnson!, geomean, chisqinvcdf,
              _loglikelihood_boxcox!,
              _loglikelihood_boxcox,
              _loglikelihood_yeojohnson!,
              _loglikelihood_yeojohnson,
              _llfunc, _llfunc!,
              _input_check,
              _centering, _scaling,
              PowerTransformation
using MixedModels: refit!

MixedModelPowerTransformation = Union{BoxCoxTransformation{LinearMixedModel},
                                      YeoJohnsonTransformation{LinearMixedModel}}

# TODO: jiggle types slightly so that this works with any powertransformation
# (maybe make PowerTransformation parametric?)
function StatsAPI.confint(t::T; level::Real=0.95,
                          fast::Bool=nobs(t) > 10_000,
                          progress=true, optimizer=:LN_COBYLA) where
         {T<:MixedModelPowerTransformation}
    lltarget = loglikelihood(t) - chisqinvcdf(1, level) / 2
    # on Julia 1.11.2-aarm64, BOBYQA seems to take us into a poorly supported
    # area of the parameter space and tests fail
    opt = NLopt.Opt(optimizer, 1)
    y = response(t)
    y_trans = similar(y)
    X = modelmatrix(t)
    ll! = _llfunc!(T)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        llhat = ll!(y_trans, X, y, only(λvec); progress)
        # want this to be zero
        val = abs(llhat - lltarget)
        return val
    end
    opt.min_objective = obj

    λ = only(params(t))
    NLopt.upper_bounds!(opt, λ)
    (ll, λvec, retval) = NLopt.optimize(opt, [λ - 1])
    lower = only(λvec)

    if fast
        upper = (λ - lower) + λ
    else
        NLopt.lower_bounds!(opt, λ)
        NLopt.upper_bounds!(opt, Inf)
        (ll, λvec, retval) = NLopt.optimize(opt, [λ + 1])
        upper = only(λvec)
    end
    return [lower, upper]
end

function StatsAPI.fit(T::Type{<:PowerTransformation}, model::LinearMixedModel;
                      progress=true,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1, kwargs...)
    isfitted(model) ||
        throw(ArgumentError("Expected model to be fitted, but `isfitted(model)` is false."))
    y = response(model)
    # we modify, so let's make a copy!
    y = (y .- _centering(T)(y)) ./ _scaling(T)(y)
    _input_check(T)(y)
    model = deepcopy(model)
    opt = NLopt.Opt(algorithm, 1)
    NLopt.xtol_abs!(opt, opt_atol) # relative criterion on parameter values
    NLopt.xtol_rel!(opt, opt_rtol) # relative criterion on parameter values
    NLopt.maxeval!(opt, maxiter)
    local y_trans = similar(y)
    ll! = _llfunc!(T)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = ll!(y_trans, model, y, only(λvec); progress)
        return val
    end
    opt.max_objective = obj
    (ll, λ, retval) = NLopt.optimize(opt, [0.0])

    return T{LinearMixedModel}(; λ=only(λ), y, X=model, kwargs...)
end

#####
##### Box Cox
#####

function BoxCox._loglikelihood_boxcox!(y_trans::Vector{<:Number}, model::LinearMixedModel,
                                       y::Vector{<:Number}, λ::Number;
                                       progress=true, kwargs...)
    _boxcox!(y_trans, y, λ; kwargs...)
    refit!(model, y_trans; progress)
    y_trans .-= fitted(model)
    return _loglikelihood_boxcox(y_trans)
end

function BoxCox._loglikelihood_boxcox(λ::Number, model::LinearMixedModel,
                                      y::Vector{<:Number};
                                      kwargs...)
    return _loglikelihood_boxcox!(similar(response(model)), model, y, λ)
end

function BoxCox._loglikelihood_boxcox(model::LinearMixedModel, y::Vector{<:Number},
                                      λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, model, y, λ[i])
    end
    return ll
end

#####
##### YeoJohnson
#####

function BoxCox._loglikelihood_yeojohnson!(y_trans::Vector{<:Number},
                                           model::LinearMixedModel,
                                           y::Vector{<:Number}, λ::Number;
                                           progress=true, kwargs...)
    _yeojohnson!(y_trans, y, λ; kwargs...)
    refit!(model, y_trans; progress)
    y_trans .-= fitted(model)
    return _loglikelihood_yeojohnson(y_trans, y, λ)
end

function BoxCox._loglikelihood_yeojohnson(λ::Number, model::LinearMixedModel,
                                          y::Vector{<:Number};
                                          kwargs...)
    return _loglikelihood_yeojohnson!(similar(response(model)), model, y, λ)
end

function BoxCox._loglikelihood_yeojohnson(model::LinearMixedModel, y::Vector{<:Number},
                                          λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_yeojohnson!(y_trans, model, y, λ[i])
    end
    return ll
end

end # module
