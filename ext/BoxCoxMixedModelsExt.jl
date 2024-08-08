module BoxCoxMixedModelsExt

using BoxCox
using MixedModels

using BoxCox: StatsAPI, NLopt
using BoxCox: _boxcox!, geomean, chisqinvcdf,
              _loglikelihood_boxcox!,
              _loglikelihood_boxcox,
              _llfunc, _llfunc!,
              _input_check,
              _scaling,
              PowerTransformation
using MixedModels: refit!

# TODO: jiggle types slightly so that this works with any powertransformation
# (maybe make PowerTransformation parametric?)
function StatsAPI.confint(bc::T; level::Real=0.95,
                          fast::Bool=nobs(bc) > 10_000, progress=true) where
                          {T<:BoxCoxTransformation{LinearMixedModel}}
    lltarget = loglikelihood(bc) - chisqinvcdf(1, level) / 2
    opt = NLopt.Opt(:LN_BOBYQA, 1)
    y_trans = similar(bc.y)
    ll! = _llfunc!(T)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        llhat = ll!(y_trans, bc.X, bc.y, only(λvec); progress)
        # want this to be zero
        val = abs(llhat - lltarget)
        return val
    end
    opt.min_objective = obj

    NLopt.upper_bounds!(opt, bc.λ)
    (ll, λvec, retval) = NLopt.optimize(opt, [bc.λ - 1])
    lower = only(λvec)

    if fast
        upper = (bc.λ - lower) + bc.λ
    else
        NLopt.lower_bounds!(opt, bc.λ)
        NLopt.upper_bounds!(opt, Inf)
        (ll, λvec, retval) = NLopt.optimize(opt, [bc.λ + 1])
        upper = only(λvec)
    end
    return [lower, upper]
end

function StatsAPI.fit(T::Type{<:PowerTransformation}, model::LinearMixedModel; progress=true,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1, kwargs...)
    isfitted(model) ||
        throw(ArgumentError("Expected model to be fitted, but `isfitted(model)` is false."))
    y = response(model)
    _input_check(T)(y)
    y = float.(y)  # we modify, so let's make a copy!
    y ./= _scaling(T)(y)
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

end # module
