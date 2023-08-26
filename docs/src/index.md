# BoxCox.jl Documentation

```@meta
CurrentModule = BoxCox
```

*BoxCox.jl* is a Julia package providing an implementation of the Box-Cox transformation and generalizations thereof.

```@example Intro
using BoxCox
using CairoMakie
using Random
```

# Box-Cox transformations of an unconditional distribution

First, we consider applying the Box-Cox transformation to an unconditional distribution.
In other words, we have a vector of data that we wish to be more Gaussian in shape.

We start with the square root of a normal distribution.

```@example Intro
x = abs2.(randn(MersenneTwister(42), 1000))
hist(x)
```

```@example Intro
qqnorm(x)
```

We fit the Box-Cox transform.
```@example Intro
bc = fit(BoxCoxTransformation, x)
```

Now that we've fit the transform, we use it like a function to transform the original data.

```@example Intro
hist(bc.(x))
```

There is also a special method for `qqnorm` provided for objects of type `BoxCoxTransformation`, which shows the QQ plot of the transformation applied to the original data.

```@example Intro
qqnorm(bc)
```

We can also generate a diagnostic plot to see how well other parameter values would have worked for normalizing the data.

```@example Intro
boxcoxplot(bc)
```

The vertical line corresponds to the final parameter estimate. 

If we specify a confidence level, then an additional horizontal line is added, which crosses the likelihood profile at the points corresponding to the edge of the confidence interval.

```@example Intro
boxcoxplot(bc; conf_level=0.95)
```
