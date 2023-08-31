# MixedModels.jl integration

```@meta
CurrentModule = BoxCox
```

BoxCox.jl supports finding fitting the Box-Cox transformation to a `LinearMixedModel` from MixedModels.jl.
On Julia 1.9 and above, this support is done via a package extension and so the user pays no dependency or precompilation penalty when this functionality is not used.
On Julia 1.6 to 1.8, this functionality is defined unconditionally (thus incurring the dependency and precompilation penalty), but neither the `@formula` macro nor the `MixedModel` type are re-exported, so MixedModels.jl must still be loaded to use this functionality.


Let us examine the classic sleepstudy dataset from MixedModels.jl. First, we load the necessary packages.

```@example Mixed
using BoxCox
using CairoMakie
using MixedModels
using MixedModels: dataset
```

Then we fit the traditional model used reaction time as our dependent variable:

```@example Mixed
model = fit(MixedModel,
            @formula(reaction ~ 1 + days + (1 + days | subj)),
            dataset(:sleepstudy))
```

While this model does perform well overall, we can also examine whether the Box-Cox transformation suggests a transformation of the response.

```@example Mixed
bc = fit(BoxCoxTransformation, model)
```

!!! note
    For large models, fitting the `BoxCoxTransformation` can take a while because a mixed model must be repeatedly fit after each intermediate transformation.

Although we receive a single "best" value (approximately -1.0747)  the fitting process, it is worthwhile to look at the profile likelihood plot for the transformation:

```@example Mixed
boxcoxplot(bc; conf_level=0.95)
```

Here we see that -1 is nearly as good. Moreover, ``\text{time}^-1`` has a natural interpretation as *speed*.
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

```@example Mixed
model_bc = fit(MixedModel,
               @formula(1 / reaction ~ 1 + days + (1 + days | subj)),
                dataset(:sleepstudy))
```

Finally, let's take a look at our the residual diagnostics for our transformed and untransformed models:

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Speed")
    density!(ax, residuals(model))
    ax = Axis(f[1, 2]; title="Reaction Time")
    density!(ax, residuals(model_bc))
    f
end
```

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1)
    qqnorm!(ax, residuals(model); qqline=:fitrobust)
    ax = Axis(f[1, 2]; title="Speed", aspect=1)
    qqnorm!(ax, residuals(model_bc); qqline=:fitrobust)
    f
end
```

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1, xlabel="Fitted", ylabel="Residual")
    scatter!(ax, fitted(model), residuals(model))
    hlines!(ax, 0; linestyle=:dash, color=:black)
    ax = Axis(f[1, 2]; title="Speed", aspect=1, xlabel="Fitted", ylabel="Residual")
    scatter!(ax, fitted(model_bc), residuals(model_bc))
    hlines!(ax, 0; linestyle=:dash, color=:black)
    f
end
```

```@example Mixed
let f = Figure()
    ax = Axis(f[1, 1]; title="Reaction Time", aspect=1)
    scatter!(ax, fitted(model), response(model), xlabel="Fitted", ylabel="Observed")
    ablines!(ax, 0, 1; linestyle=:dash, color=:black)
    ax = Axis(f[1, 2]; title="Speed", aspect=1, xlabel="Fitted", ylabel="Observed")
    scatter!(ax, fitted(model_bc), response(model_bc))
    ablines!(ax, 0, 1; linestyle=:dash, color=:black)
    f
end
```

All together, this suggests that using speed instead of reaction time does indeed improve the quality of the model fit, even though the fit was already very good for reaction time.
This example also highlighted the importance of the analyst's discretion: we choose a slightly different transformation than the originally estimated Box-Cox transformation in order to yield a model on a naturally interpretable scale.