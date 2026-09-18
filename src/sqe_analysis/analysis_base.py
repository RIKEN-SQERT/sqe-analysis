"""
Abstract base classes for data analysis

For concrete classes implementing the analysis, see the
:py:mod:`~sqe_analysis.analysis` module.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, cast, override

import xarray as xr

from sqe_analysis.result import (
    AnalysisResult,
    CurvefitAnalysisResult,
    get_source_dataset_id,
)


class BaseAnalysis(ABC):
    """
    Base class that defines the API for all kinds of data analysis.

    There is only one function that a subclass should implement, :py:meth:`run`.
    It takes in an Xarray DataArray and returns an
    :py:class:`~sqe_analysis.result.AnalysisResult` object. See the
    documentation of that class for further information. See also :doc:`the
    tutorial for creating a custom data analysis class
    </guide/creating-analyses>`.

    Having a class with a single method may seem a bit redundant, but the goal
    is to keep the interface consistent across subclasses (such as
    :py:class:`CurvefitAnalysis`) which may need more methods.

    Note that ``run`` is a class method, so it cannot depend on any internal
    state.
    """

    @classmethod
    @abstractmethod
    def run(cls, data: xr.DataArray, *args: Any, **kwargs: Any) -> AnalysisResult:
        """
        Perform data analysis.

        The keyword arguments should contain additional parameters needed for
        the analysis, such as the dimension(s) over which to do curve fitting.
        """
        raise NotImplementedError(
            f"Analysis not implemented for {cls.__module__}.{cls.__qualname__}"
        )


CurvefitCoordsType = str | xr.DataArray | Iterable[str | xr.DataArray]
"""
Type of ``coords`` in ``CurvefitAnalysis.run()``.

See `the Xarray curvefit documentation <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.curvefit.html>`_
for more information.
"""


CurvefitGuessType = Mapping[str, float | xr.DataArray]
"""
The type of initial guess of xr.DataArray.curvefit, mapping from string to float
or DataArray
"""

CurvefitBoundsType = Mapping[
    str,
    tuple[float | xr.DataArray, float | xr.DataArray],
]
"""
Parameter bounds for xr.DataArray.curvefit.

Maps parameter names to pairs of lower and upper bounds. Each bound can
be a scalar or a DataArray.
"""


class CurvefitAnalysis(BaseAnalysis):
    """
    Special case of analysis where the analysis is performed by fitting a curve.

    A subclass should implement the model function by overriding
    :py:meth:`func`. The :py:meth:`run` method has a default implementation that
    performs fitting to ``func`` using ``xr.DataArray.curvefit`` and returns a
    :py:class:`~sqe_analysis.result.CurvefitAnalysisResult`. Additionally, a
    subclass may implement :py:meth:`guess` for initial values,
    :py:meth:`bounds` for default parameter bounds, and
    :py:meth:`preprocess` for simple transformations before fitting.

    This class should only be used for the cases where the analysis truly
    consists of a single curve fit. If you need to perform multiple curve fits
    (for example, fitting an oscillation frequency as a function of some
    parameter, and then fitting a curve to the extracted oscillation
    frequencies), you should use :py:class:`BaseAnalysis` instead, and use
    ``CurvefitAnalysis`` subclasses in the ``run`` implementation. See :ref:`the
    tutorial <multi-step-analysis>` for an example.

    Note that all methods are class methods, so they cannot depend on any
    internal state.
    """

    # TODO: doc link to xarray curvefit

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
    def guess(
        cls,
        preprocessed_data: xr.DataArray,
        coords: CurvefitCoordsType,
    ) -> CurvefitGuessType | None:
        """
        Initial guess for the curve fitting.

        Note that this function is called after preprocessing.

        The return value should have the same format as the ``p0`` parameter of
        ``xr.DataArray.curvefit``, i.e. a dictionary that maps parameter names
        to scalar values or data arrays (if the initial guess varies with a
        coordinate of multidimensional data).

        Returns ``None`` if an initial guess is not implemented.
        """
        # TODO: link to xarray docs
        return None

    @classmethod
    def bounds(cls) -> CurvefitBoundsType | None:
        """
        Default parameter bounds for curve fitting.

        Returns a mapping from parameter names to tuples of ``(lower, upper)`` bounds,
        in the same format as the ``bounds`` parameter of ``xr.DataArray.curvefit``.

        Returns ``None`` if no default bounds are defined.
        """
        return None

    @classmethod
    def preprocess(
        cls,
        data: xr.DataArray,
        coords: CurvefitCoordsType,
    ) -> xr.DataArray | None:
        """
        Apply simple preprocessing to the data before fitting or generating an
        initial guess.

        The preprocessing should be simple enough that it does **not** affect
        the main quantities of interest. A common preprocessing step is
        :py:func:`~sqe_analysis.signal_processing.project_complex`.

        The result of this function will be stored in
        ``.intermediate_results.preprocessed_data`` of the fit result object.

        Returns ``None`` if no preprocessing is to be performed.
        """
        # TODO: add more explicit concrete examples of what kinds of
        # transformations are acceptable and which are not, once we have more
        # concrete examples of fitting
        return None

    @classmethod
    @override
    def run(
        cls,
        data: xr.DataArray,
        coords: CurvefitCoordsType,
        guess: CurvefitGuessType | None = None,
        bounds: CurvefitBoundsType | None = None,
        curvefit_kwargs: dict[str, Any] | None = None,
    ) -> CurvefitAnalysisResult:
        """
        Analyze data by performing curve fitting.

        This is a thin wrapper around the `Xarray curvefit <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.curvefit.html>`__
        function.

        Initial guesses are passed to xarray after merging automatic and explicit
        guesses. If an initial guess is outside the effective bounds, a ValueError
        is raised, as in xarray and SciPy.

        Args:
            data: Data to analyze
            coords: Coordinate(s) of the data along which to perform curve fitting.
            guess: Parameter values for initial guess. These will override any
                parameters returned by :py:meth:`guess`.
            curvefit_kwargs: Keyword arguments passed to `xr.DataArray.curvefit`.
            bounds: Parameter bounds overriding the output of :py:meth:`bounds` for
                the specified parameters. ``None`` or an empty mapping gives no overrides.
                Use ``(-np.inf, np.inf)`` to remove the bounds for a parameter.

        Raises:
            ValueError: If an initial guess is outside the effective bounds.
        """
        # TODO: automatically determine coords? longest dim? and separate subclass for 2D fit with 2 longest coords?

        if guess is None:
            guess = {}

        if curvefit_kwargs is None:
            curvefit_kwargs = {}

        # some default values for curvefit_kwargs
        curvefit_kwargs = dict(
            {
                "errors": "ignore",
            },
            **curvefit_kwargs,
        )

        preprocessed_data = cls.preprocess(data, coords=coords)
        if preprocessed_data is not None:
            data_to_fit = preprocessed_data
        else:
            data_to_fit = data

        guess_from_func = cls.guess(data_to_fit, coords=coords)
        if guess_from_func is not None:
            guess = {**guess_from_func, **guess}

        bounds_from_func = cls.bounds()
        merged_bounds = {
            **(bounds_from_func or {}),
            **(bounds or {}),
        }

        # TODO: Consider clipping initial guesses to be within bounds.
        fit_result = data_to_fit.curvefit(
            coords=coords,
            func=cls.func,
            p0=guess,
            bounds=merged_bounds,
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
        fit_params_guess = None
        if guess:
            fit_params_guess = xr.Dataset(guess)

        fit_params = cast(
            xr.Dataset,
            fit_result.curvefit_coefficients.to_dataset("param"),
        )
        result_params = fit_params
        extra_params = cls.extra_params(result_params)
        if extra_params is not None:
            result_params = result_params.merge(extra_params)

        # If errors='ignore' in curvefit_kwargs, the coefficients and
        # covariances for the coordinates where the fitting failed will be NaN.
        # TODO: add option for success based on chisquare threshold or similar?
        # TODO: add possibility to manually define success based on fit
        # parameters, e.g. if qubit frequency is negative or similar
        success = ~fit_result.curvefit_coefficients.isel(param=0, drop=True).isnull()

        return CurvefitAnalysisResult(
            # TODO: consider possibility of excluding some of the fit parameters
            # from params (e.g. with a `exlcude_params` attribute).
            params=result_params,
            fit_params=fit_params,
            success=success,
            # TODO: convert curvefit covariances to std (or store full covariance matrix???)
            # params_std=fit_result.curvefit_covariances.to_dataset("param"),
            intermediate_results=intermediate_results,
            fit_params_guess=fit_params_guess,
            analysis_class=cls,
            source_dataset_id=get_source_dataset_id(data),
        )

    # TODO: 'fixed' method that returns a copy of cls where 'func' is wrapped
    # such that some of the parameters have fixed values

    @classmethod
    def extra_params(cls, fit_params: xr.Dataset) -> xr.Dataset | None:
        """
        Additional quantities of intereset derived from the fit results that are
        not parameters of the model function.

        Args:
            fit_params: The result of the curve fitting. It is a Dataset with
                a data variable for each fit parameter

        Returns:
            An Xarray Dataset with data variables for each of the additional
            derived quantities, or ``None`` if there are no derived quantities.
        """
        # TODO: link to xarray dataset docs
        return None
