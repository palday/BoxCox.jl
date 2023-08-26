module BoxCoxStatsModelsExt

using BoxCox
using BoxCox: StatsAPI
using StatsModels
using Tables


function StatsAPI.fit(::Type{BoxCoxTransformation}, f::FormulaTerm, data;
                     contrasts=Dict{Symbol,Any}(), kwargs...)
    tbl = Tables.columntable(data)
    fvars = StatsModels.termvars(f)
    tvars = Tables.columnnames(tbl)
    fvars ⊆ tvars ||
        throw(
            ArgumentError(
                "The following formula variables are not present in the table: $(setdiff(fvars, tvars))",
            ),
        )

    # TODO: perform missing_omit() after apply_schema() when improved
    # missing support is in a StatsModels release
    tbl, _ = StatsModels.missing_omit(tbl, f)
    sch = schema(f, tbl, contrasts)
    form = apply_schema(f, sch, RegressionModel)
    y, X = modelcols(form, tbl)
    return fit(BoxCoxTransformation, X, y; kwargs...)
end

end # module
