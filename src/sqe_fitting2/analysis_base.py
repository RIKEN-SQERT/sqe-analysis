"""
Abstract base classes for data analysis
"""

from collections.abc import Iterable, Mapping
from typing import Any

import xarray as xr


class BaseAnalysis:
    """
    Base class that defines the API for all kinds of data analysis.

    The main function a subclass should implement is `analyze`, which takes in
    an Xarray DataArray (that has been possibly preprocessed by `preprocess`)
    and returns an Xarray Dataset with the following schema: **TOD**

    The main function that a user should use is `run`, which will call
    `preprocess` (if it is implemented) and then `analyze`.

    Note that all methods are class methods, so they cannot depend on any
    internal state.
    """

    @classmethod
    def preprocess(cls, data: xr.DataArray) -> xr.DataArray | None:  # pyright: ignore [reportUnusedParameter]
        """
        Any preprocessing that should be done to the raw data before performing
        the analysis.

        If no preprocessing is done, this function should return None, which is
        the default behavior. This way, `run` can determine whether to save the
        preprocessed data in the result dataset.
        """
        return None

    @classmethod
    def analyze(
        cls, preprocessed_data: xr.DataArray, *args: Any, **kwargs: Any
    ) -> xr.Dataset:
        """
        The main analysis function. Should return a dataset with a specified
        schema, see the class documentation.

        The keyword arguments should contain additional parameters needed for
        the analysis, such as the dimension(s) over which to do curve fitting.
        """
        raise NotImplementedError(
            f"Analysis not implemented for {cls.__module__}.{cls.__qualname__}"
        )

    @classmethod
    def run(cls, data: xr.DataArray, **kwargs: Any) -> xr.Dataset:
        """
        Preprocess and then analyze data.

        The preprocessed data is saved in the result dataset so that it can be
        used for e.g. plotting. If the `preprocess` function returns None, it is
        not added to the dataset.
        """
        preprocessed = cls.preprocess(data)

        data_to_analyze = data if preprocessed is None else preprocessed

        analysis_result = cls.analyze(data_to_analyze, **kwargs)

        if preprocessed is not None:
            analysis_result = analysis_result.assign(preprocessed_data=preprocessed)

        return analysis_result


# The type of initial guess of xr.DataArray.curvefit
CurvefitGuessType = Mapping[str, float | xr.DataArray]


class CurvefitAnalysis(BaseAnalysis):
    """
    Special case of analysis where the analysis is performed by fitting a curve.

    A subclass should implement the model function in `func`. The `analyze`
    function has a default implementation that performs fitting to `func` using
    `xr.DataArray.curvefit`. Additionally, a subclass may implement a `guess`
    function that produces an initial guess, which will be called by `analyze`.

    This class should only be used for the cases where the analysis truly
    consists of a single curve fit. If you need to perform multiple curve fits
    (for example, fitting an oscillation frequency as a function of some
    parameter, and then fitting a curve to the extracted oscillation
    frequencies), you should use `BaseAnalysis` instead, and use
    `CurvefitAnalysis` subclasses in the `analyze` implementation. See the
    ???documentation??? for an example.

    Note that all methods are class methods, so they cannot depend on any
    internal state.
    """

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
    def analyze(
        cls,
        preprocessed_data: xr.DataArray,
        coords: str | xr.DataArray | Iterable[str | xr.DataArray],
        guess: CurvefitGuessType | None = None,
        curvefit_kwargs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        """
        Analysis based on curve fitting.

        Args:
            curvefit_kwargs: Keyword arguments passed to `xr.DataArray.curvefit`.
        """
        # TODO: automatically determine coords? longest dim? and separate subclass for 2D fit with 2 longest coords?
        # TODO: bounds

        if guess is None:
            guess = {}

        if curvefit_kwargs is None:
            curvefit_kwargs = {}

        # TODO: merge guess and guess_from_func
        _guess_from_func = cls.guess(preprocessed_data)

        return preprocessed_data.curvefit(
            coords=coords,
            func=cls.func,
            p0=guess,
            **curvefit_kwargs,
        )

    # TODO: 'fixed' method that returns a copy of cls where 'func' is wrapped
    # such that some of the parameters have fixed values
