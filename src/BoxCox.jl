module BoxCox

using LinearAlgebra
using NLopt
using Statistics
using StatsAPI
using StatsBase
using Printf

export BoxCoxTransformation,
       loglikelihood,
       response,
       is_fitted,
       fitted,
       fit,
       predict

abstract type PowerTransformation end
# struct BoxCoxTransformation <: PowerTransformation end
# struct YeoJohnsonTransformation <: PowerTransformation end
# struct BickelDoksumTransformation <: PowerTransformation end

# params, offset, nobs, confint

BoxCoxTransformation = @NamedTuple begin
    λ::Float64 # power
    α::Float64 # shift
    normalization::Float64 # generally the geometric mean; does not apply to log
    y::Vector{Float64} # observed response
    atol::Float64 # isapprox tolerance
end

function _boxcox(λ, x; α=0, normalization=1)
    x += α
    λ == 0 && return normalization * log(x)
    return (x^λ - 1) / (λ * normalization^(λ - 1))
end

function (t::BoxCoxTransformation)(x::Number)
    return _boxcox(t.λ, x; t.α, t.normalization)
end

# fit! ?
# should we support passing a formula or model matrix?

function StatsAPI.fit(::Type{BoxCoxTransformation}, x::AbstractVector{Float64};
                      algorithm::Symbol=:LN_NELDERMEAD, atol=1e-8, rtol=1e-8, maxiter=-1)
    # lowerbound for precision is 0, everything else has no lowerbound
    opt = NLopt.Opt(algorithm, 1)
    NLopt.xtol_rel!(opt, atol) # relative criterion on parameter values
    NLopt.xtol_rel!(opt, rtol) # relative criterion on parameter values
    NLopt.maxeval!(opt, maxiter)
    function obj(λvec, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = _loglikelihood(only(λvec), 0, 1, x)
        return val
    end
    opt.max_objective = obj
    (ll, λ, retval) = optimize(opt, [0.0])
    return BoxCoxTransformation((only(λ), 0, 1, x))
end

StatsAPI.fitted(t::BoxCoxTransformation) = predict(t, response(t))
StatsAPI.predict(t::BoxCoxTransformation, v::AbstractVector{<:Number}) = t.(v)

# pull this out so that we can use it in optimization
function _loglikelihood(λ, x, α=0, normalization=1)
    tx = _boxcox(λ, x; α, normalization)
    n = length(tx)
    σ² = var(tx; corrected=false)

    return -n / 2 * log(σ²) + (t.λ - 1) * sum(log, x)
end

function StatsAPI.loglikelihood(t::BoxCoxTransformation,
                                x::AbstractVector{Float64}=response(t))
    return _loglikelihood(t.λ, x; t.α, t.normalization)
end

StatsAPI.response(t::BoxCoxTransformation) = t.y

function Base.show(io::IO, t::BoxCoxTransformation)
    println(io, "Box-Cox transformation")
    println(io)
    if !isapprox(t.α, 0)
        @printf io "a priori α:    %.2f\n" t.α
    end
    if !isapprox(t.normalization, 1)
        @printf io "normalization: %.2f\n" t.normalization
    end
    @printf io "\nestimated λ: %.2f" t.λ
    println(io, "\nresultant transformation:\n")

    if isapprox(t.λ, 1)
        println(io, "y (the identity)")
        return nothing
    end

    λ = @sprintf "%.1f" t.λ
    norm = @sprintf "%.1f" t.normalization
    α = @sprintf "%.1f" t.α

    if isapprox(0, t.λ)
        result = isapprox(t.α, 0) ? "log y" : "log(y + $(α))"
        if !isapprox(t.normalization, 1)
            result = string(norm, " ", result)
        end
        println(io, result)
        return nothing
    end

    numerator = isapprox(t.α, 0) ? "y^$(λ) - 1" : "(y + $(α))^$(λ) - 1"
    denominator = λ
    if !isapprox(t.normalization, 1)
        denominator *= string(" * ", norm, @sprintf("^%.1f", t.λ - 1))
    end
    width = maximum(length, [numerator, denominator]) + 2

    println(io, lpad(numerator, (width - length(numerator)) ÷ 2 + length(numerator)))
    println(io, "-"^width)
    println(io, lpad(denominator, (width - length(denominator)) ÷ 2 + length(denominator)))

    return nothing
end
end # module BoxCox
