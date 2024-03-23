# MixedModels.jl integration

```@meta
CurrentModule = BoxCox
```

BoxCox.jl supports finding fitting the Box-Cox transformation to a `LinearMixedModel` from MixedModels.jl.
This support is done via a package extension and so the user pays no dependency or precompilation penalty when this functionality is not used.


Let us examine the classic sleepstudy dataset from MixedModels.jl. First, we load the necessary packages.

```@example Mixed
using BoxCox
using CairoMakie
using MixedModels
using MixedModels: dataset
CairoMakie.activate!(; type="svg")
```

Then we fit the traditional model used reaction time as our dependent variable:

```@example Mixed
model = fit(MixedModel,
            @formula(reaction ~ 1 + days + (1 + days | subj)),
            dataset(:sleepstudy))
```


## Fitting the Box-Cox transformation

While this model does perform well overall, we can also examine whether the Box-Cox transformation suggests a transformation of the response.

```@example Mixed
bc = fit(BoxCoxTransformation, model)
```

!!! note
    For large models, fitting the `BoxCoxTransformation` can take a while because a mixed model must be repeatedly fit after each intermediate transformation.


## Choosing an appropriate transformation

Although we receive a single "best" value (approximately -1.0747) from the fitting process, it is worthwhile to look at the profile likelihood plot for the transformation:

```@example Mixed
boxcoxplot(bc; conf_level=0.95)
```

Here we see that -1 is nearly as good. Moreover, time``^{-1}`` has a natural interpretation as *speed*.
In other words, we can model reaction speed instead of reaction time.
Then instead of seeing whether participants take longer to respond with each passing day, we can see whether their speed increases or decreases.
In both cases, we're looking at whether they respond *faster* or *slower* and even the terminology *fast* and *slow* suggests that speed is easily interpretable.

Now, the formal definition of the Box-Cox transformation is:

```math
\begin{cases}
\frac{x^{\lambda} - 1}{\lambda} &\quad \lambda \neq 0 \\
\log x &\quad \lambda = 0
\end{cases}
```

In other words, there is a normalizing denominator that flips the sign when ``\lambda < 0``.
If we use the full Box-Cox formula, then the sign of the effect in our transformed and untransformed model remains the same.
While useful at times, speed has a natural interpretation and so we instead use the power relation, which is the actual key component, without normalization.


## Fitting a model to the transformed response

Because `reaction` is stored in milliseconds, we use `1000 / reaction` instead of `1 / reaction` so that our speed units are responses per second.

```@example Mixed
model_bc = fit(MixedModel,
               @formula(1000 / reaction ~ 1 + days + (1 + days | subj)),
                dataset(:sleepstudy))
```

For our original model on the untransformed scale, the intercept was approximately 250, which means that the average response time was about 250 milliseconds.
For the model on the speed scale, we have an intercept about approximately 4, which means that the average response speed is about 4 responses per second, which implies that the the average response time is 250 milliseconds.
In other words, our new results are compatible with our previous estimates.

!!! note
    Because the Box-Cox transformation helps a model achieve normality of the *residuals*, it helps fulfill the model assumptions. When these assumptions are not fulfilled, we may still get similar estimates, but the standard errors and derived measures (e.g., confidence intervals and associated coverage) may not be correct.

Finally, let's take a look at our the residual diagnostics for our transformed and untransformed models:

## Impact of transformation

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1)
    density!(ax, residuals(model))
    ax = Axis(f[1, 2]; title="Speed", aspect=1)
    density!(ax, residuals(model_bc))
    colsize!(f.layout, 1, Aspect(1, 1.0))
    colsize!(f.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(f)
    f
end
```

### QQ plots

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1)
    qqnorm!(ax, residuals(model); qqline=:fitrobust)
    ax = Axis(f[1, 2]; title="Speed", aspect=1)
    qqnorm!(ax, residuals(model_bc); qqline=:fitrobust)
    colsize!(f.layout, 1, Aspect(1, 1.0))
    colsize!(f.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(f)
    f
end
```

### Fitted vs residual

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1, xlabel="Fitted", ylabel="Residual")
    scatter!(ax, fitted(model), residuals(model))
    hlines!(ax, 0; linestyle=:dash, color=:black)
    ax = Axis(f[1, 2]; title="Speed", aspect=1, xlabel="Fitted", ylabel="Residual")
    scatter!(ax, fitted(model_bc), residuals(model_bc))
    hlines!(ax, 0; linestyle=:dash, color=:black)
    colsize!(f.layout, 1, Aspect(1, 1.0))
    colsize!(f.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(f)
    f
end
```

### Fitted vs observed

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1)
    scatter!(ax, fitted(model), response(model), xlabel="Fitted", ylabel="Observed")
    ablines!(ax, 0, 1; linestyle=:dash, color=:black)
    ax = Axis(f[1, 2]; title="Speed", aspect=1, xlabel="Fitted", ylabel="Observed")
    scatter!(ax, fitted(model_bc), response(model_bc))
    ablines!(ax, 0, 1; linestyle=:dash, color=:black)
    colsize!(f.layout, 1, Aspect(1, 1.0))
    colsize!(f.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(f)
    f
end
```

All together, this suggests that using speed instead of reaction time does indeed improve the quality of the model fit, even though the fit was already very good for reaction time.
This example also highlighted the importance of the analyst's discretion: we choose a slightly different transformation than the originally estimated Box-Cox transformation in order to yield a model on a naturally interpretable scale.
