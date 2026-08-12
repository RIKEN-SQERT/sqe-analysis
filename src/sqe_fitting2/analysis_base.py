"""
Abstract base classes for data analysis
"""

from collections.abc import Iterable, Mapping
from typing import Any, override

import xarray as xr

from sqe_fitting2.result import AnalysisResult, CurvefitAnalysisResult


class BaseAnalysis:
    """
    Base class that defines the API for all kinds of data analysis.

    There is only one function that a subclass should implement, `run`. It takes
    in an Xarray DataArray and returns an Xarray Dataset with the following
    schema: **TOD**

    Having a class with a single method may seem a bit redundant, but the goal
    is to keep the interface consistent across subclasses (such as
    `CurvefitAnalysis`) which may need more methods.

    Note `run` is a class method, so it cannot depend on any internal state.
    """

    @classmethod
    def run(cls, data: xr.DataArray, *argd: Any, **kwargs: Any) -> AnalysisResult:
        """
        Perform data analysis.

        Should return a dataset with a specified schema, see the class documentation.

        The keyword arguments should contain additional parameters needed for
        the analysis, such as the dimension(s) over which to do curve fitting.
        """
        raise NotImplementedError(
            f"Analysis not implemented for {cls.__module__}.{cls.__qualname__}"
        )


# The type of initial guess of xr.DataArray.curvefit
CurvefitGuessType = Mapping[str, float | xr.DataArray]


class CurvefitAnalysis(BaseAnalysis):
    """
    Special case of analysis where the analysis is performed by fitting a curve.

    A subclass should implement the model function in `func`. The `run` function
    has a default implementation that performs fitting to `func` using
    `xr.DataArray.curvefit`. Additionally, a subclass may implement a `guess`
    function that produces an initial guess, which will be called by `run`.

    This class should only be used for the cases where the analysis truly
    consists of a single curve fit. If you need to perform multiple curve fits
    (for example, fitting an oscillation frequency as a function of some
    parameter, and then fitting a curve to the extracted oscillation
    frequencies), you should use `BaseAnalysis` instead, and use
    `CurvefitAnalysis` subclasses in the `run` implementation. See the
    ???documentation??? for an example.

    Note that all methods are class methods, so they cannot depend on any
    internal state.
    """

    # TODO: document preprocessing...

    # TODO: doc link

    @classmethod
    def func(cls, *independent_vars: Any, **kwargs: Any):
        """
        The model function used for curve fitting. The independent variable(s)
        (e.g. 'time' or 'x') should be the first argument(s).
        """
        raise NotImplementedError(
            f"Model function not implemented for {cls.__module__}.{cls.__qualname__}"
        )

    @classmethod
    def guess(cls, preprocessed_data: xr.DataArray) -> CurvefitGuessType | None:
        """
        Initial guess for the curve fitting.

        Note that this function is called after preprocessing.

        The return value should have the same format as the `p0` parameter of
        `xr.DataArray.curvefit`, i.e. a dictionary that maps parameter names to
        scalar values or data arrays (if the initial guess varies with a
        coordinate of multidimensional data).
        """
        return None

    @classmethod
    def preprocess(
        cls,
        data: xr.DataArray,
        coords: str | xr.DataArray | Iterable[str | xr.DataArray],
    ) -> xr.DataArray | None:
        # TODO: docstrirng
        # ... should return a data array that is suitable for fitting to the model ...
        return None

    @classmethod
    @override
    def run(
        cls,
        data: xr.DataArray,
        coords: str | xr.DataArray | Iterable[str | xr.DataArray],
        guess: CurvefitGuessType | None = None,
        curvefit_kwargs: dict[str, Any] | None = None,
    ) -> CurvefitAnalysisResult:
        """
        Analysis based on curve fitting.

        Args:
            curvefit_kwargs: Keyword arguments passed to `xr.DataArray.curvefit`.
            guess: Parameter values for initial guess. These will override any
                parameters returned by `cls.guess`.
        """
        # TODO: automatically determine coords? longest dim? and separate subclass for 2D fit with 2 longest coords?
        # TODO: bounds

        if guess is None:
            guess = {}

        if curvefit_kwargs is None:
            curvefit_kwargs = {}

        preprocessed_data = cls.preprocess(data, coords=coords)
        if preprocessed_data is not None:
            data_to_fit = preprocessed_data
        else:
            data_to_fit = data

        guess_from_func = cls.guess(data_to_fit)
        if guess_from_func is not None:
            # override from guess provided as argument
            guess = {**guess_from_func, **guess}

        fit_result = data_to_fit.curvefit(
            coords=coords,
            func=cls.func,
            p0=guess,
            **curvefit_kwargs,
        )

        intermediate_results = {}

        # TODO: consider making preprocessed data store ptional
        if preprocessed_data is not None:
            intermediate_results["preprocessed_data"] = preprocessed_data

        # TODO: maybe we should allow dict as argument of AnalysisResult and
        # convert to dataset there, so we don't have to do this
        if not intermediate_results:
            intermediate_results = None
        else:
            intermediate_results = xr.Dataset(intermediate_results)

        # TODO: this is similar to the intermediate_results case above but inconsistent...
        params_guess = None
        if guess:
            params_guess = xr.Dataset(guess)

        return CurvefitAnalysisResult(
            params=fit_result.curvefit_coefficients.to_dataset("param"),
            # TODO: convert curvefit covariances to std (or store full covariance matrix???)
            # params_std=fit_result.curvefit_covariances.to_dataset("param"),
            intermediate_results=intermediate_results,
            params_guess=params_guess,
        )

    # TODO: 'fixed' method that returns a copy of cls where 'func' is wrapped
    # such that some of the parameters have fixed values
