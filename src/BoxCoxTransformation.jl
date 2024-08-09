"""
    BoxCoxTransformation <: PowerTransformation

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
    boxcox(λ; atol=0)
    boxcox(λ, x; atol=0)

Compute the Box-Cox transformation of x for the parameter value λ.

The Box-Cox transformation is defined as:

```math
\\begin{cases}
\\frac{x^{\\lambda} - 1}{\\lambda} &\\quad \\lambda \\neq 0 \\\\
\\log x &\\quad \\lambda = 0
\\end{cases}
```

for positive ``x``. (If ``x <= 0``, then ``x`` must first be translated to be strictly positive.)

`atol` controls the absolute tolerance for treating λ as zero.

The one argument variant curries and creates a one-argument function of `x` for the given λ.

See also [`BoxCoxTransformation`](@ref).

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

See also [`boxcox`](@ref).
"""
function (t::BoxCoxTransformation)(x::Number)
    return boxcox(t.λ, x)
end

"""
    boxcoxplot(bc::PowerTransformation; kwargs...)
    boxcoxplot!(axis::Axis, bc::PowerTransformation;
                λ=nothing, n_steps=21, xlabel="λ", ylabel="log likelihood",
                conf_level=nothing, attributes...)

Create a diagnostic plot for the Box-Cox transformation.

The mutating method for `Axis` returns the (modified) original `Axis`.
The non-mutating method returns a `Figure`.

If λ is `nothing`, the range of possible values for the λ parameter is automatically determined,
with a total of `n_steps`. If `λ` is a vector of numbers, then the λ parameter is evaluated at
each element of that vector.

If `conf_level` is `nothing`, then no confidence interval is displayed.

`attributes` are forwarded to `scatterlines!`.

!!! note
    A meaningful plot is only possible when `bc` has not been `empty!`'ed.

!!! note
    The plotting functionality interface is defined as a package extension and only loaded when Makie is available.
    You must load an appropriate Makie backend (e.g., CairoMakie or GLMakie) to actually render a plot.
"""
function boxcoxplot!(::Any, ::Any; kwargs...)
    # specialize slightly so that they can't just throw Any and get this message
    throw(ArgumentError("Have you loaded an appropriate Makie backend?"))
end

"$(@doc boxcoxplot!)"
function boxcoxplot(::Any; kwargs...)
    throw(ArgumentError("Have you loaded an appropriate Makie backend?"))
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

function Base.show(io::IO, t::BoxCoxTransformation)
    println(io, "Box-Cox transformation")
    @printf io "\nestimated λ: %.4f" t.λ
    # if !isempty(response(t))
    #     println(io, "\np-value: ", StatsBase.PValue(pvalue(t)))
    # end
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

#####
##### Internal methods that traits redirect to
#####

# used in Makie extension
function _loglikelihood_boxcox(X::AbstractMatrix{<:Number}, y::Vector{<:Number},
                               λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    Xqr = qr(X)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, Xqr, X, y, λ[i])
    end
    return ll
end

# used in Makie extension
function _loglikelihood_boxcox(::Nothing, y::Vector{<:Number},
                               λ::AbstractVector{<:Number})
    y_trans = similar(y)
    ll = similar(λ)
    for i in eachindex(ll, λ)
        ll[i] = _loglikelihood_boxcox!(y_trans, y, λ[i])
    end
    return ll
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

function _input_check_boxcox(y)
    any(<=(0), y) && throw(ArgumentError("all y values must be greater than zero"))
    return nothing
end

#####
##### Traits
#####

# need the <: to handle the parameterized type

_input_check(::Type{<:BoxCoxTransformation}) = _input_check_boxcox
_llfunc(::Type{<:BoxCoxTransformation}) = _loglikelihood_boxcox
_llfunc!(::Type{<:BoxCoxTransformation}) = _loglikelihood_boxcox!
_centering(::Type{<:BoxCoxTransformation}) = Returns(0)
_scaling(::Type{<:BoxCoxTransformation}) = geomean
