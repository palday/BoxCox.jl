# BoxCox.jl Documentation

```@meta
CurrentModule = BoxCox
```

*BoxCox.jl* is a Julia package providing an implementation of the Box-Cox transformation and generalizations thereof.


# Box-Cox transformations of an unconditional distribution

First, we consider applying the Box-Cox transformation to an unconditional distribution.
In other words, we have a vector of data that we wish to be more Gaussian in shape.

We start with the square root of a normal distribution.

```@example Unconditional
using BoxCox
using CairoMakie
using Random
x = abs2.(randn(MersenneTwister(42), 1000))
hist(x)
```

```@example Unconditional
qqnorm(x)
```

We fit the Box-Cox transform.
```@example Unconditional
bc = fit(BoxCoxTransformation, x)
```

Note that the resulting transform isn't exactly a square root, even though our data are just the square of a normal sample. The reason for this is simple: without knowing the original sign, the square root returns all positive values and would thus not generate a symmetric distribution. The Box-Cox transformation does not eliminate the use of the analyst's discretion and domain knowledge.

Now that we've fit the transform, we use it like a function to transform the original data.

```@example Unconditional
hist(bc.(x))
```

There is also a special method for `qqnorm` provided for objects of type `BoxCoxTransformation`, which shows the QQ plot of the transformation applied to the original data.

```@example Unconditional
qqnorm(bc)
```

We can also generate a diagnostic plot to see how well other parameter values would have worked for normalizing the data.

```@example Unconditional
boxcoxplot(bc)
```

The vertical line corresponds to the final parameter estimate. 

If we specify a confidence level, then an additional horizontal line is added, which crosses the likelihood profile at the points corresponding to the edge of the confidence interval.

```@example Unconditional
boxcoxplot(bc; conf_level=0.95)
```

# Box-Cox transformations of a conditional distribution

We can also consider transformations of a conditional distribution.
As an example, we consider the `trees` dataset:

```@example Conditional
using BoxCox
using CairoMakie
using RDatasets: dataset as rdataset
using StatsModels

trees = rdataset("datasets", "trees")
```

For the conditional distribution, we want to fit a linear regression to the transformed response values and then evaluate the profile likelihood of the transformation. If the StatsModels package has been loaded, either directly or indirectly (e.g. via loading GLM.jl or MixedModels.jl), then a formula interface is available. (Otherwise, the model matrix and the response have to specified separately.)

```@example Conditional
bc = fit(BoxCoxTransformation, @formula(Volume ~ log(Height) + log(Girth)), trees)
```

We can do all the same diagnostics as previously:

```@example Conditional
let f = Figure()
    ax = Axis(f[1, 1]; title="Raw")
    hist!(ax, trees.Volume)
    ax = Axis(f[1, 2]; title="Transformed")
    hist!(ax, bc.(trees.Volume))
    f
end
```

```@example Conditional
let f = Figure()
    ax = Axis(f[1, 1]; title="Raw", aspect=1)
    qqnorm!(ax, trees.Volume; qqline=:fitrobust)
    ax = Axis(f[1, 2]; title="Transformed", aspect=1)
    qqnorm!(ax, bc)
    f
end
```

```@example Conditional
boxcoxplot(bc; conf_level=0.95)
```

This last diagnostic plot suggests that λ = 0 is within the possible range of parameter values to consider.  λ = 0 corresponds to a logarithmic transformation; given that the other variables are log-transformed, this suggests that we should consider using a log transform for the response. 
