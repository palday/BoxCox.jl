module BoxCox

using Compat
using LinearAlgebra
using NLopt
using Printf
using Statistics
using StatsAPI
using StatsBase
using StatsFuns

export BoxCoxTransformation,
       loglikelihood,
       fit,
       confint,
       nobs,
       boxcoxplot,
       boxcoxplot!

abstract type PowerTransformation end
# struct BoxCoxTransformation <: PowerTransformation end
# struct YeoJohnsonTransformation <: PowerTransformation end
# struct BickelDoksumTransformation <: PowerTransformation end

@kwdef struct BoxCoxTransformation <: PowerTransformation
    λ::Float64 # power
    y::Vector{Float64} # observed response normalized by its geometric mean
    X::Union{Nothing,Matrix{Float64}}
    atol::Float64 = 1e-8 # isapprox tolerance to round towards zero or one
end

function Base.isapprox(x::BoxCoxTransformation, y::BoxCoxTransformation; kwargs...)
    return isapprox(x.λ, y.λ; kwargs...)
end

function boxcox(λ, x; atol=0)
    if isapprox(λ, 0; atol)
        logx = log(x)
        λlogx = λ * logx
        return logx * (1 + (λlogx) / 2 * (1 + (λlogx) / 3 * (1 + (λlogx) / 4)))
    end

    return (x^λ - 1) / λ
end

function (t::BoxCoxTransformation)(x::Number)
    return boxcox(t.λ, x)
end

# stop carrying around the data
function Base.empty!(bt::BoxCoxTransformation)
    empty!(bt.y)
    empty!(bt.X)
    return bt
end

Base.isempty(bt::BoxCoxTransformation) = any(isempty, [bt.y, bt.x])

function StatsAPI.fit(::Type{BoxCoxTransformation}, y::AbstractVector{<:Number}; atol=1e-8,
                      algorithm::Symbol=:LN_BOBYQA, opt_atol=1e-8, opt_rtol=1e-8,
                      maxiter=-1)
    y = convert(Vector{Float64}, y)
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

function _boxcox!(y_trans, y, λ)
    for i in eachindex(y, y_trans)
        y_trans[i] = boxcox(λ, y[i])
    end
    return y_trans
end

# pull this out so that we can use it in optimization
function _loglikelihood_boxcox!(y_trans::Vector{<:Number}, Xqr::Factorization,
                                X::Matrix{<:Number}, y::Vector{<:Number}, λ::Number)
    _boxcox!(y_trans, y, λ)
    y_trans -= X * (Xqr \ y_trans)
    return _loglikelihood_boxcox(y_trans)
end

function _loglikelihood_boxcox!(y_trans::Vector{<:Number}, y::Vector{<:Number}, λ::Number)
    _boxcox!(y_trans, y, λ)
    y_trans .-= mean(y_trans)
    return _loglikelihood_boxcox(y_trans)
end

function _loglikelihood_boxcox(y_trans::Vector{<:Number})
    return -0.5 * length(y_trans) * log(sum(abs2, y_trans))
end

function _loglikelihood_boxcox(λ::Number, X::AbstractMatrix{<:Number}, y::Vector{<:Number})
    return _loglikelihood_boxcox!(similar(y), qr(X), X, y, λ)
end

function _loglikelihood_boxcox(λ::Number, ::Nothing, y::Vector{<:Number})
    return _loglikelihood_boxcox!(similar(y), y, λ)
end

# th = 0.670280702687321

StatsAPI.nobs(bc::BoxCoxTransformation) = length(bc.y)
function _pvalue(bc::BoxCoxTransformation)
    llhat = loglikelihood(bc)
    ll0 = _loglikelihood_boxcox(0, bc.X, bc.y)
    return chisqcdf(1, 2 * (llhat - ll0))
end

# function StatsAPI.confint(bc::BoxCoxTransformation)
#     ll0 = _loglikelihood_boxcox(0, bc.X, bc.y)

#     lltarget = loglikelihood(bc) - chisqinvcdf(1, 0.95) / 2
#     opt = NLopt.Opt(:LN_COBYLA, 1)
#     NLopt.upper_bounds!(opt, bc.λ)
#     function obj(λvec, g)
#         isempty(g) || throw(ArgumentError("g should be empty for this objective"))
#         llhat = _loglikelihood_boxcox(only(λvec), bc.X, bc.y)
#         return min(llhat, lltarget)
#     end
#     opt.max_objective = obj
#      (ll, λd, retval) = optimize(opt, [bc.λ - 1])
#     return [only(λd), bc.λ + (bc.λ - only(λd))]
# end

_sse(bc::BoxCoxTransformation) = _sse(bc.λ, bc.X, bc.y)
_sse(λ::Number, bc::BoxCoxTransformation) = _sse(λ, bc.X, bc.y)

function _sse(λ::Number, X::AbstractMatrix{<:Number}, y::Vector{<:Number})
    y_trans = similar(y)
    _boxcox!(y_trans, y, λ)
    y_trans -= X * (Xqr \ y_trans)
    return y_trans'y_trans
end

function _sse(λ::Number, ::Nothing, y::Vector{<:Number})
    y_trans = similar(y)
    _boxcox!(y_trans, y, λ)
    y_trans .-= mean(y_trans)
    return y_trans'y_trans
end

StatsAPI.loglikelihood(t::BoxCoxTransformation) = _loglikelihood_boxcox(t.λ, t.X, t.y)

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

# specialize slightly so that they can't just throw Any and get this message
function boxcoxplot!(::Any, ::PowerTransformation; kwargs...)
    throw(ArgumentError("Have you loaded an appropriate Makie backend?"))
end

function boxcoxplot(::PowerTransformation; kwargs...)
    throw(ArgumentError("Have you loaded an appropriate Makie backend?"))
end

if !isdefined(Base, :get_extension)
    include("../ext/BoxCoxStatsModelsExt.jl")
    include("../ext/BoxCoxMakieExt.jl")
end

end # module BoxCox
