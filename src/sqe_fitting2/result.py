import xarray as xr
from xarray.core.datatree import DatasetView


class AnalysisResult:
    """
    A thin wrapper around an Xarray DataTree.

    This wrapper exists so that we can do type-checking and autocomplete on the
    datasets stored in the DataTree, and verify that they follow the result
    schema.

    Args:
        params: The qantities of interest of the analysis. The Dataset should
            have one data variable for each parameter.
        params_std: Estimates of the standard deviations of the extracted
            parameters, if available.
        intermediate_results: Additional analysis results that are needed for
            visualization, but not direct quantities of interest of the analysis
        debug_results: Additional analysis results that are useful for
            troubleshooting the analysis, but not necessary for visualization.
            These may not be saved when ??saving the result as netcdf??.
    """

    def __init__(
        self,
        params: xr.Dataset,
        params_std: xr.Dataset | None = None,
        intermediate_results: xr.Dataset | None = None,
        debug_results: xr.Dataset | None = None,
    ):
        children: dict[str, xr.DataTree] = {"params": xr.DataTree(params)}
        if params_std is not None:
            children["params_std"] = xr.DataTree(params_std)
        if intermediate_results is not None:
            children["intermediate_results"] = xr.DataTree(intermediate_results)
        if debug_results is not None:
            children["debug_results"] = xr.DataTree(debug_results)
        self._data: xr.DataTree = xr.DataTree(children=children)

    @property
    def params(self) -> DatasetView:
        # TODO: docstring...
        return self._data.params.dataset

    @property
    def params_std(self) -> DatasetView | None:
        node = self._data.get("params_std")
        return node.dataset if node is not None else None

    @property
    def intermediate_results(self) -> DatasetView | None:
        node = self._data.get("intermediate_results")
        return node.dataset if node is not None else None

    @property
    def debug_results(self) -> DatasetView | None:
        node = self._data.get("debug_results")
        return node.dataset if node is not None else None

    def to_netcdf(self, path: ...):
        raise NotImplementedError

    @classmethod
    def from_netcdf(cls, path: ...):
        # TODO: option / function to validate the schema
        raise NotImplementedError

    def _repr_html_(self):
        return self._data._repr_html_().replace(
            "xarray.DataTree", f"{self.__class__.__name__} (xarray.DataTree)"
        )


class CurvefitAnalysisResult(AnalysisResult):
    """
    Specialized analysis result for curve fitting-based analysis.

    Args:
        params: The quantities of interest of the analysis. These may be
            parameters of the model function, but may also contain additional
            derived quantities, and may not necessarily contain all model
            function parameters if they are not of interest. To evaluate the
            model function, use `fit_params` instead.
        fit_params: The parameters of the model function that minimize the error
            between the model prediction and the data. These should always
            exactly match the arguments of the model function, so that they can
            be used to evaluate it.
        fit_params_guess: The initial guess used for the fitting, either
            calculated using the `guess` method or explicitly passed as
            arguments. May not contain all parameters of the model function, so
            it might not be possible to directly evaluate it with the guess
            values.
    """

    def __init__(
        self,
        params: xr.Dataset,
        fit_params: xr.Dataset,
        params_std: xr.Dataset | None = None,
        intermediate_results: xr.Dataset | None = None,
        fit_params_guess: xr.Dataset | None = None,
        debug_results: xr.Dataset | None = None,
    ):
        super().__init__(
            params=params,
            params_std=params_std,
            intermediate_results=intermediate_results,
            debug_results=debug_results,
        )
        data = self._data.assign(fit_params=xr.DataTree(fit_params))
        if fit_params_guess is not None:
            data = data.assign(fit_params_guess=xr.DataTree(fit_params_guess))
        self._data = data

    @property
    def fit_params(self) -> DatasetView:
        """
        Dataset containing the parameters of the model function that minimize
        the error between the model prediction and the data. This can be used to
        evaluate the model function:

            >>> result = MyCurvefitAnalysis.run(data, dim="x")
            >>> fit_eval = MyCurvefitAnalysis.func(data.x, **result.fit_params)
        """
        return self._data.fit_params.dataset

    @property
    def fit_params_guess(self) -> DatasetView | None:
        node = self._data.get("fit_params_guess")
        return node.dataset if node is not None else None
