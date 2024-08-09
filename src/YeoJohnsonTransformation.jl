# See also [`boxcoxplot`](@ref), [`params`](@ref), [`yeojohnson`](@ref).

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
    "The original response"
    y::Vector{Float64}
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

_identity(yt::YeoJohnsonTransformation) = YeoJohnsonTransformation(; yt.y, λ=1, yt.X, yt.atol)

function Base.show(io::IO, t::YeoJohnsonTransformation)
    println(io, "Yeo-Johnson transformation")
    @printf io "\nestimated λ: %.4f" t.λ
    if !isempty(response(t))
        println(io, "\np-value: ", StatsBase.PValue(pvalue(t)))
    end
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

# used in Makie extension
function _loglikelihood_yeojohnson(X::AbstractMatrix{<:Number}, y::Vector{<:Number},
                                   λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    Xqr = qr(X)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_yeojohnson!(y_trans, Xqr, X, y, λ[i])
    end
    return ll
end

# used in Makie extension
function _loglikelihood_yeojohnson(::Nothing, y::Vector{<:Number},
                                   λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_yeojohnson!(y_trans, y, λ[i])
    end
    return ll
end

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
    return _loglikelihood_yeojohnson(y_trans, y, λ)
end

# setup marginal distribution
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

_input_check_yeojohnson(::Any) = nothing

#####
##### Traits
#####

# need the <: to handle the parameterized type

_input_check(::Type{<:YeoJohnsonTransformation}) =_input_check_yeojohnson
_llfunc(::Type{<:YeoJohnsonTransformation}) = _loglikelihood_yeojohnson
_llfunc!(::Type{<:YeoJohnsonTransformation}) = _loglikelihood_yeojohnson!
_scaling(::Type{<:YeoJohnsonTransformation}) = Returns(1)
_centering(::Type{<:YeoJohnsonTransformation}) = Returns(0)
