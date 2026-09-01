"""
Classes for storing analysis results
"""

import warnings
from dataclasses import dataclass
from datetime import datetime

import xarray as xr


@dataclass(kw_only=True)
class AnalysisResult:
    """
    Data class for storing analysis results.

    All attributes are Xarray Datasets so that their values can be
    multidimensional arrays.
    """

    # TODO: validate that params_std is a subset of params (and the coordinates match)
    # TODO: also verify that the coordinates of success match with params

    params: xr.Dataset
    """
    The qantities of interest of the analysis. The Dataset should have one data
    variable for each parameter.
    """

    params_std: xr.Dataset | None = None
    """
    Estimates of the standard deviations of the extracted
    parameters, if available. May contain only a subset of `params`.
    """

    success: xr.DataArray
    """
    A DataArray containing booleans indicating whether the analysis was a
    success. The coordinates should be the same as those of `params`.
    """

    analysis_class: str | type
    """
    Name of the data analysis class that produced this result. Can also be the
    class itself, but will be converted to a string.
    """

    source_dataset_id: str
    """
    Unique identifier of the data set that this analysis result was derived
    from.
    """

    intermediate_results: xr.Dataset | None = None
    """
    Additional analysis results that are needed for visualization, but not
    direct quantities of interest of the analysis
    """

    debug_results: xr.Dataset | None = None
    """
    Additional analysis results that are useful for troubleshooting the
    analysis, but not necessary for visualization. These may not be saved when
    ??saving the result as netcdf??.
    """

    created_at: str = ""
    """
    ISO 8601 timestamp of when this result was created. Will be filled in
    automatically upon initialization.
    """

    def to_netcdf(self, path: ...):
        raise NotImplementedError

    @classmethod
    def from_netcdf(cls, path: ...):
        # TODO: option / function to validate the schema
        raise NotImplementedError

    # TODO: reimplement this after changing from DataTree wrapper to dataclass
    def _repr_html_(self):
        # TODO: proper implementation. Now this is just a hack to piggy-back on DataTree repr
        from dataclasses import fields

        children = {}
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if v is not None:
                # success is a DataArray, convert it to dataset
                if isinstance(v, xr.DataArray):
                    v = xr.Dataset({f.name: v})
                elif isinstance(v, str):
                    continue
                children[k] = xr.DataTree(v)
        dt = xr.DataTree(children=children)
        # as of xarray 2026.7.0, there is no assign_attrs method on DataTree so just assign manually
        dt.attrs["analysis_class"] = self.analysis_class
        dt.attrs["source_dataset_id"] = self.source_dataset_id
        dt.attrs["created_at"] = self.created_at
        return dt._repr_html_().replace("xarray.DataTree", f"{self.__class__.__name__}")

    def __post_init__(self):
        # ensure that analysis_class is a string
        if isinstance(self.analysis_class, type):
            self.analysis_class = (
                f"{self.analysis_class.__module__}.{self.analysis_class.__qualname__}"
            )

        self.created_at = datetime.now().astimezone().isoformat()


@dataclass(kw_only=True)
class CurvefitAnalysisResult(AnalysisResult):
    """
    Specialized analysis result for curve fitting-based analysis.
    """

    # TODO: validate that the parameters that are present both in params and
    # fit_params are equal
    # TODO: example that shows how to evaluate the model function with `fit_params`

    fit_params: xr.Dataset
    """
    Dataset containing the parameters of the model function that minimize the
    error between the model prediction and the data. These should always exactly
    match the arguments of the model function, so that they can be used to
    evaluate it.
    """

    # TODO: fit_params_std for the uncerainties of the fit parameters

    fit_params_guess: xr.Dataset | None = None
    """
    The initial guess used for the fitting, either calculated using the `guess`
    method or explicitly passed as arguments. May not contain all parameters of
    the model function, so it might not be possible to directly evaluate it with
    the guess values.
    """


def get_source_dataset_id(data: xr.DataArray) -> str:
    """
    Helper function to check that the data has 'dataset_id' in the attributes.
    Issues a warning if the dataset ID is missing.
    """
    k = "dataset_id"
    default = "unknown"
    if k not in data.attrs:
        warnings.warn(
            f"{k!r} not found in dataset attributes, defaulting to {default!r}. The dataset ID should be defined.",
            stacklevel=3,
        )
        return default
    else:
        return data.attrs[k]
