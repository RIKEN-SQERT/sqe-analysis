from dataclasses import dataclass

import xarray as xr


@dataclass(kw_only=True)
class AnalysisResult:
    """
    Data class for storing analysis results.

    All attributes are Xarray Datasets so that their values can be
    multidimensional arrays.

    Attributes:
        params: The qantities of interest of the analysis. The Dataset should
            have one data variable for each parameter.
        params_std: Estimates of the standard deviations of the extracted
            parameters, if available. May contain only a subset of `params`.
        intermediate_results: Additional analysis results that are needed for
            visualization, but not direct quantities of interest of the analysis
        debug_results: Additional analysis results that are useful for
            troubleshooting the analysis, but not necessary for visualization.
            These may not be saved when ??saving the result as netcdf??.
    """

    params: xr.Dataset
    params_std: xr.Dataset | None = None
    intermediate_results: xr.Dataset | None = None
    debug_results: xr.Dataset | None = None

    def to_netcdf(self, path: ...):
        raise NotImplementedError

    @classmethod
    def from_netcdf(cls, path: ...):
        # TODO: option / function to validate the schema
        raise NotImplementedError

    # TODO: reimplement this after changing from DataTree wrapper to dataclass
    # def _repr_html_(self):
    #    return self._data._repr_html_().replace(
    #        "xarray.DataTree", f"{self.__class__.__name__} (xarray.DataTree)"
    #    )


@dataclass(kw_only=True)
class CurvefitAnalysisResult(AnalysisResult):
    """
    Specialized analysis result for curve fitting-based analysis.

    Attributes:
        fit_params: Dataset containing the parameters of the model function that
            minimize the error between the model prediction and the data. These
            should always exactly match the arguments of the model function, so
            that they can be used to evaluate it.
        fit_params_guess: The initial guess used for the fitting, either
            calculated using the `guess` method or explicitly passed as
            arguments. May not contain all parameters of the model function, so
            it might not be possible to directly evaluate it with the guess
            values.
    """

    # TODO: validate that the parameters that are present both in params and
    # fit_params are equal
    # TODO: example that shows how to evaluate the model function with `fit_params`

    fit_params: xr.Dataset

    # TODO: fit_params_std for the uncerainties of the fit parameters

    fit_params_guess: xr.Dataset | None = None
