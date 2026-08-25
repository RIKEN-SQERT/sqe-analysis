"""
The main API of the library.

The analysis classes are ordered alphabetically, for lack of better organization.
"""

from typing import cast, override

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from xarray.core.types import Dims

from sqe_fitting2.analysis_base import BaseAnalysis
from sqe_fitting2.result import AnalysisResult, CurvefitAnalysisResult
from sqe_fitting2.signal_processing import project_complex
from sqe_fitting2.xr_util import longest_dim


class ExponentialRegressionAnalysis(BaseAnalysis):
    """
    Analysis for exponentially decaying data with non-zero baseline

    Fits the model

        a * exp(-k * x) + b

    to the data. Note that this is not a subclass of CurvefitAnalysis. The
    paramateres are directly extracted from the data without any curve fitting
    or guessing, based on the method described by J. Jacquelin, see
    https://stackoverflow.com/a/39436209,
    https://math.stackexchange.com/a/1337641,
    https://www.scribd.com/document/23155389/Theoretical-Impedance-of-Capacitive-Electrodes,
    https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales (in
    French). Note that the parameter names are different from those used by
    Jacquelin.

    This method has the advantage that it is extremely fast and requires no initial
    guess. The downside is that it does not provide error bounds for the
    parameters. It can be used as an initial guess for curve fitting.
    """

    # TODO: link & math formatting in docstring (also run() docstring)
    # TODO: example showing that it also works for complex-valued data

    @classmethod
    @override
    def run(
        cls, data: xr.Dataset, dim: str | None = None, snr_threshold: float = 5.0
    ) -> CurvefitAnalysisResult:
        """
        Run the analysis.

        Args:
            data: The data to analyze
            dim: The dimension along which the exponential decay occurs, usually
                time. If None, use the longest dimesnsion of the data.
            snr_threshold: Threshold for SNR below which the result is marked as
                a failure. If `dim` is `t`, the SNR is calculated as

                    SNR = |f(t=0) - f(t=t_max)| / |std(data - f(t))|,

                where f(t) is the model function evaluated with the extracted
                parameters, and t_max is the maximum value of `dim` in the data.
                The standard deviation `std` is calculated over `dim`.

        Returns:
            A `CurvefitAnalysisResult` (even though we are not doing curve
            fitting), with `fit_params` corresponding to the exponential
            parameters. `params` contains an additional parameter for the
            decay constant, which is the inverse of the scale factor `k` in
            the exponent.
        """
        # Comparsion of naming convention of Jacquelin:
        # | us | Jacquelin |
        # |  a | b         |
        # |  b | a         |
        # |  k | -c        |

        if dim is None:
            dim = longest_dim(data)

        y = data
        x = y[dim]
        # normalize coordinate to avoid numerical instability with big or small numbers
        x_range = (x.max() - x.min()).item()
        y = y.assign_coords({dim: y[dim] / x_range})
        x = y[dim]

        # following https://stackoverflow.com/a/39436209

        # S_k, it's the trapezoidal rule
        s = y.cumulative_integrate(dim)

        # Shifted coordinates relative to the first point
        xx0 = x - x.isel({dim: 0})
        yy0 = y - y.isel({dim: 0})

        # elements of the first matrix
        m1_00 = (xx0**2).sum(dim)
        m1_01 = (xx0 * s).sum(dim)  # same as m1_10
        m1_11 = (s**2).sum(dim)

        # first vector
        v1_0 = (yy0 * xx0).sum(dim)
        v1_1 = (yy0 * s).sum(dim)

        # the matrix is 2x2 so we can invert it analytically and explicitly
        # compute the matrix-vector product of the bottom row (the first row is
        # not used)
        det1 = m1_00 * m1_11 - m1_01**2
        c = (-m1_01 * v1_0 + m1_00 * v1_1) / det1

        theta = cast(xr.DataArray, np.exp(c * x))

        m2_00 = x.size  # n
        m2_01 = theta.sum(dim)  # same as m2_10
        m2_11 = (theta**2).sum(dim)

        v2_0 = y.sum(dim)
        v2_1 = (y * theta).sum(dim)

        det2 = m2_00 * m2_11 - m2_01**2

        # first row of inverse (note inverted naming of a & b)
        b = (m2_11 * v2_0 - m2_01 * v2_1) / det2
        # second row of inverse
        a = (m2_00 * v2_1 - m2_01 * v2_0) / det2

        fit_params = xr.Dataset(
            {
                "a": a,
                "b": b,
                # we have to reverse the normalization here
                "k": -c / x_range,
            }
        )

        signal = abs(
            cls.func(0, **fit_params) - cls.func(data[dim].max(), **fit_params)
        )
        noise = (data - cls.func(data[dim], **fit_params)).std()
        snr = signal / noise

        return CurvefitAnalysisResult(
            params=fit_params.assign(
                # TODO: add SNR
                decay_constant=1 / fit_params.k,
                SNR=snr,
            ),
            success=snr > snr_threshold,
            fit_params=fit_params,
        )

    @classmethod
    def func(cls, x: ArrayLike, k, a, b) -> ArrayLike:
        """
        Exponential model function.
        """
        return a * np.exp(-k * x) + b


class TimeOfFlightAnalysis(BaseAnalysis):
    """
    Analysis for time-of-flight measurement

    Assumes that the data is demodulated.

    Always returns a result that is a value from the time axis of the dataset.

    Returns:
        A Dataset with the variable ???

    Example:

        >>> from sqe_fitting2.example_data import open_dataset
        >>> y = open_dataset("")

        Quick visualization of the result:
    """

    @classmethod
    @override
    def run(
        cls,
        data: xr.DataArray,
        dim: Dims | None = None,
        snr_threshold: float = 2.0,
        smoothing: int = 5,
    ) -> AnalysisResult:
        # TODO: how should I split project_complex vs smoothing in preprocess??
        if dim is None:
            dim = longest_dim(data)

        proj = project_complex(data, dim=dim)
        smoothed = proj.rolling({dim: smoothing}, center=True).mean()
        diff = smoothed.differentiate(dim)

        # TODO: proper peak finding...
        step_locations = abs(diff).idxmax(dim)
        pre_step = proj.where(diff[dim] < step_locations)
        post_step = proj.where(diff[dim] > step_locations)
        signal = abs(post_step.median([dim]) - pre_step.median([dim]))
        noise = pre_step.std([dim])
        snr = signal / noise
        return AnalysisResult(
            params=xr.Dataset(
                dict(  # noqa: C408
                    step_location=step_locations,
                    SNR=snr,
                ),
            ),
            success=snr > snr_threshold,
            intermediate_results=xr.Dataset(
                dict(  # noqa: C408
                    data_projected=proj,
                )
            ),
            debug_results=xr.Dataset(
                dict(  # noqa: C408
                    smoothed_diff=diff,
                )
            ),
        )
