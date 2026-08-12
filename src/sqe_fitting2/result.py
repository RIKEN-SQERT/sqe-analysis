import xarray as xr
from xarray.core.datatree import DatasetView


class AnalysisResult:
    """
    A thin wrapper around an Xarray DataTree.

    This wrapper exists so that we can do type-checking on the datasets stored
    in the DataTree and verify that they follow the result schema.

    Args:
        params: The qantities of interest of the analysis. The Dataset should
            have one data variable for each parameter.
        params_std: Estimates of the standard deviations of the extracted
            parameters, if available.
        intermediate_results: Additional analysis results that are needed for
            visualization, but not direct quantities of interest of the analysis
        debug_results: Additional analysis results that are useful for troubleshooting the analysis, but
    """

    def __init__(
        self,
        params: xr.Dataset,
        params_std: xr.Dataset | None = None,
        intermediate_results: xr.Dataset | None = None,
        debug_results: xr.Dataset | None = None,
    ):
        self._data: xr.DataTree = xr.DataTree(
            children=dict(  # noqa: C408
                params=xr.DataTree(params),
                params_std=xr.DataTree(params_std),
                intermediate_results=xr.DataTree(intermediate_results),
                debug_results=xr.DataTree(debug_results),
            )
        )

    @property
    def params(self) -> DatasetView:
        # TODO: docstring...
        return self._data.params.dataset

    @property
    def params_std(self) -> DatasetView | None:  # TODO: check None return type...
        return self._data.params_std.dataset

    @property
    def intermediate_results(self) -> DatasetView | None:
        return self._data.intermediate_results.dataset

    @property
    def debug_results(self) -> DatasetView | None:
        return self._data.debug_results.dataset

    def to_netcdf(self, path: ...):
        raise NotImplementedError

    def from_netcdf(self, path: ...):
        # TODO: option / function to validate the schema
        raise NotImplementedError

    def _repr_html_(self):
        return self._data._repr_html_().replace(
            "xarray.DataTree", f"{self.__class__.__name__} (xarray.DataTree)"
        )
