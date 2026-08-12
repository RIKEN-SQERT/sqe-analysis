import xarray as xr
from dataclasses import dataclass


#@dataclass(frozen=True) # TODO: is dataclass appropriate
class AnalysisResult:
    """
    A thin wrapper around an Xarray DataTree.
    """
    def __init__(
        self,
        params: xr.Dataset,
        params_std: xr.Dataset | None = None,
        intermediate_results: xr.Dataset | xr.DataTree | None = None,
        debug_results: xr.Dataset | xr.DataTree | None = None,
    ):
        self._data = xr.DataTree(
            children=dict(
                params=xr.DataTree(params),
                params_std=xr.DataTree(params_std),
                intermediate_results=xr.DataTree(intermediate_results),
                debug_results=xr.DataTree(debug_results),
            )
        )

    @property
    def params(self) -> xr.Dataset:
        # TODO: docstring...
        return self._data.params.dataset

    @property
    def params_std(self) -> xr.Dataset | None:
        return self._data.params_std.dataset
    
    @property
    def intermediate_results(self) -> xr.DatasetView:
        return self._data.intermediate_results

    @property
    def debug_results(self) -> xr.DatasetView:
        return self._data.intermediate_results

    def to_netcdf(self):
        raise NotImplementedError

    def from_netcdf(self, path: ...):
        # TODO: function to validate the schema
        raise NotImplementedError
    
    def _repr_html_(self):
        return self._data._repr_html_()
