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
yeojohnson(λ; kwargs...) = Base.Fix1(λ)
function yeojohnson(λ, x; atol=0)
    if y >= 0
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


# StatsAPI.confint -- can re refactor boxcox slightly so that we can share an implementaiton?
# Base.show

#####
##### Internal methods that traits redirect to
#####

function _loglikelihood_yeojohnson!(y_trans::Vector{<:Number}, Xqr::Factorization,
                                X::Matrix{<:Number}, y::Vector{<:Number}, λ::Number;
                                kwargs...)
    _yeojohnson!(y_trans, y, λ; kwargs...)
    y_trans -= X * (Xqr \ y_trans)
    return _loglikelihood_yeojohnson(y_trans)
end

function _loglikelihood_yeojohnson!(y_trans::Vector{<:Number}, y::Vector{<:Number}, λ::Number;
                                kwargs...)
    _yeojohnson!(y_trans, y, λ; kwargs...)
    y_trans .-= mean(y_trans)
    return _loglikelihood_yeojohnson(y_trans)
end

function _loglikelihood_yeojohnson(y_trans::Vector{<:Number})
    return -0.5 * length(y_trans) * log(sum(abs2, y_trans))
end

function _loglikelihood_yeojohnson(λ::Number, X::AbstractMatrix{<:Number}, y::Vector{<:Number};
                               kwargs...)
    return _loglikelihood_yeojohnson!(similar(y), qr(X), X, y, λ)
end

function _loglikelihood_yeojohnson(λ::Number, ::Nothing, y::Vector{<:Number}; kwargs...)
    return _loglikelihood_yeojohnson!(similar(y), y, λ)
end

# no domain restrictions here besides real values 😎
_input_check_yeojohnson(::Any) = nothing

#####
##### Traits
#####

# need the <: to handle the parameterized type

_input_check(::Type{<:YeoJohnsonTransformation}) =_input_check_yeojohnson
_llfunc(::Type{<:YeoJohnsonTransformation}) = _loglikelihood_yeojohnson
_llfunc!(::Type{<:YeoJohnsonTransformation}) = _loglikelihood_yeojohnson!
# XXX should this be identity???
_scaling(::Type{<:YeoJohnsonTransformation}) = geomean
