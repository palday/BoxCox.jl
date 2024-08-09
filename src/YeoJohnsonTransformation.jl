# See also [`boxcoxplot`](@ref), [`params`](@ref), [`boxcox`](@ref).

"""
    YeoJohnsonTransformation <: PowerTransformation

# Fields

$(FIELDS)

!!! note
    All fields are considered internal and implementation details and may change at any time without
    being considered breaking.

# Tips

- To extract the λ parameter, use `params`.
- The transformation is callable, meaning that you can do
```@example
yj = fit(YeoJohnsonTransformation, y)
y_transformed = yj.(y)
```
- You can reduce the size of a YeoJohnsonTransformation in memory by using `empty!`, but certain diagnostics
  (e.g. plotting and computation of the loglikelihood will no longer be available).

"""
Base.@kwdef struct YeoJohnsonTransformation{T} <: PowerTransformation
    "The transformation parameter"
    λ::Float64
    "The original response, normalized by its geometric mean"
    y::Vector{Float64} # observed response normalized by its geometric mean
    "A model matrix for the conditional distribution or `Nothing` for the unconditional distribution "
    X::T
    "Tolerance for comparing λ to zero or two. Default is 1e-8"
    atol::Float64 = 1e-8 # isapprox tolerance to round towards zero or one
end

function YeoJohnsonTransformation(λ::Number, y::Vector, X::T, atol::Number) where {T}
    return YeoJohnsonTransformation{T}(; λ, y, X, atol)
end

"""
    yeojohnson(λ; atol=0)
    yeojohnson(λ, x; atol=0)

```math
    yeojohnson(λ; atol=0)
    yeojohnson(λ, x; atol=0)

\\begin{cases} ((x_i+1)^\\lambda-1)/\\lambda                      &  \\text{if }\\lambda \\neq 0, y \\geq 0 \\\\
               \\log(y_i + 1)                                     &  \\text{if }\\lambda =     0, y \\geq 0 \\\\
               -((-x_i + 1)^{(2-\\lambda)} - 1) / (2 - \\lambda)  &  \\text{if }\\lambda \\neq 2, y <     0 \\\\
               -\\log(-x_i + 1)                                   &  \\text{if }\\lambda =     2, y <     0
\\end{cases}
```

`atol` controls the absolute tolerance for treating λ as zero or two.

The one argument variant curries and creates a one-argument function of `x` for the given λ.

See also [`YeoJohnsonTransformation`](@ref).

# References
Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. Biometrika, 87(4), 954–959. https://doi.org/10.1093/biomet/87.4.954
"""
yeojohnson(λ; kwargs...) = x -> yeojohnson(λ, x; kwargs...)
function yeojohnson(λ, x; atol=0)
    if x >= 0
        if !isapprox(λ, 0; atol)
            ((x + 1)^λ - 1) / λ
        else
            return log1p(x)
        end
    else
        if !isapprox(λ, 2; atol)
            twoλ = (2 - λ)
            -((-x + 1)^twoλ - 1) / twoλ
        else
            return -log1p(-x)
        end
    end
end

"""
    (t::YeoJohnsonTransformation)(x::Number)

Apply the estimated YeoJohnson transformation `t` to the number `x`.

See also [`yeojohnson`](@ref).
"""
function (t::YeoJohnsonTransformation)(x::Number)
    return yeojohnson(t.λ, x)
end

# TODO: plots

"""
    _yeojohnson!(y_trans, y, λ; kwargs...)

Internal method to compute `yeojohnson` at each element of `y` and store the result in `y_trans`.
"""
function _yeojohnson!(y_trans, y, λ; kwargs...)
    for i in eachindex(y, y_trans; kwargs...)
        y_trans[i] = yeojohnson(λ, y[i])
    end
    return y_trans
end


# StatsAPI.confint -- can re refactor boxcox slightly so that we can share an implementation?

function Base.show(io::IO, t::YeoJohnsonTransformation)
    println(io, "Yeo-Johnson transformation")
    @printf io "\nestimated λ: %.4f" t.λ
    # println(io, "\np-value: ", StatsBase.PValue(_pvalue(t)))
    println(io, "\nresultant transformation:\n")

    if isapprox(t.λ, 1; t.atol)
        println(io, "y (the identity)")
        return nothing
    end

    λ = @sprintf "%.1f" t.λ

    println(io, "For y ≥ 0,\n")
    if isapprox(0, t.λ; t.atol)
        println(io, "log(y + 1)")
    else
        numerator = "(y + 1)^$(λ) - 1"
        denominator = λ
        width = maximum(length, [numerator, denominator]) + 2
        println(io, lpad(numerator, (width - length(numerator)) ÷ 2 + length(numerator)))
        println(io, "-"^width)
        println(io, lpad(denominator, (width - length(denominator)) ÷ 2 + length(denominator)))
    end
    println(io)
    println(io)

    println(io, "For y < 0:\n")
    if isapprox(2, t.λ; t.atol)
        println(io, "-log(-y + 1) for y < 0")
    else
        twoλ = "(2 - $(λ))"
        numerator = "-((-y + 1)^$(twoλ) - 1)"
        denominator = twoλ
        width = maximum(length, [numerator, denominator]) + 2
        println(io, lpad(numerator, (width - length(numerator)) ÷ 2 + length(numerator)))
        println(io, "-"^width)
        println(io, lpad(denominator, (width - length(denominator)) ÷ 2 + length(denominator)))
    end

    return nothing
end

#####
##### Internal methods that traits redirect to
#####

# setup linear regression
function _loglikelihood_yeojohnson(λ::Number, X::AbstractMatrix{<:Number}, y::Vector{<:Number};
                                   kwargs...)
    return _loglikelihood_yeojohnson!(similar(y), qr(X), X, y, λ)
end

# do linear regression
function _loglikelihood_yeojohnson!(y_trans::Vector{<:Number}, Xqr::Factorization,
                                X::Matrix{<:Number}, y::Vector{<:Number}, λ::Number;
                                kwargs...)
    _yeojohnson!(y_trans, y, λ; kwargs...)
    y_trans -= X * (Xqr \ y_trans)
    return _loglikelihood_yeojohnson(y_trans, λ)
end

# setup marginal distrbution
function _loglikelihood_yeojohnson(λ::Number, ::Nothing, y::Vector{<:Number}; kwargs...)
    return _loglikelihood_yeojohnson!(similar(y), y, λ)
end

# do marginal distribution
function _loglikelihood_yeojohnson!(y_trans::Vector{<:Number}, y::Vector{<:Number}, λ::Number;
                                kwargs...)
    _yeojohnson!(y_trans, y, λ; kwargs...)
    return _loglikelihood_yeojohnson(y_trans, y, λ)
end

# actual likelihood computation
function _loglikelihood_yeojohnson(y_trans::Vector{<:Number}, y::Vector{<:Number}, λ::Number)
    n = length(y_trans)
    σ² = var(y_trans; corrected=false)
    penalty = (λ - 1) * sum(y) do x
        return copysign(log1p(abs(x)), x)
    end
    return -0.5 * n * (1 + log2π + log(σ²)) + penalty
end

# plants = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0,  3.0, 9.3, 7.5, -6.0]
# λ = 1.305, μ = 4.570, σ² = 29.876, lrt compared to lamba=1 is, 3.873 p=0.0499
# yt0 = YeoJohnsonTransformation(; λ=1, X=nothing, y=plants)
# yt1 = YeoJohnsonTransformation(; λ=1.305, X=nothing, y=plants)
# yt0.(plants) ≈ plants
# isapprox(mean(yt1.(plants)), 4.570; atol=0.005)
# isapprox(var(yt1.(plants); corrected=false), 29.876; rtol=0.005)
# lrt = 2 * abs(loglikelihood(yt0) - loglikelihood(yt1))
# isapprox(lrt, 3.873; rtol=0.005)
# no domain restrictions here besides real values 😎
_input_check_yeojohnson(::Any) = nothing

#####
##### Traits
#####

# need the <: to handle the parameterized type

_input_check(::Type{<:YeoJohnsonTransformation}) =_input_check_yeojohnson
_llfunc(::Type{<:YeoJohnsonTransformation}) = _loglikelihood_yeojohnson
_llfunc!(::Type{<:YeoJohnsonTransformation}) = _loglikelihood_yeojohnson!
# XXX should this be geomean like boxcox???
_scaling(::Type{<:YeoJohnsonTransformation}) = identity
