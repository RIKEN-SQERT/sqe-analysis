"""
The main API of the library

The analysis classes are ordered alphabetically, for lack of better organization.
"""

from typing import Any, cast, override

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from xarray.core.types import Dims

from sqe_analysis.analysis_base import (
    BaseAnalysis,
    CurvefitAnalysis,
    CurvefitCoordsType,
    CurvefitGuessType,
)
from sqe_analysis.result import (
    AnalysisResult,
    CurvefitAnalysisResult,
    get_source_dataset_id,
)
from sqe_analysis.signal_processing import project_complex, simple_dft
from sqe_analysis.xarray_util import longest_dim


class DampedOscillationAnalysis(CurvefitAnalysis):
    r"""
    Curve fit for exponentially damped oscillations

    Fits the model

    .. math::

        b + a \cdot \exp(-x / \tau) \cdot \cos\left(2\pi (f x + \phi)\right)

    to real-valued data. Complex readout IQ is projected to the real axis
    using :py:func:`~sqe_analysis.signal_processing.project_complex`
    in :py:meth:`preprocess`.

    For supported inputs, :py:meth:`guess` estimates initial values for all
    model parameters. Values supplied through the ``guess`` argument of
    :py:meth:`run` override these estimates.

    For complex input, ``a``, ``b``, and ``phi`` describe the projected
    signal. The projection subtracts the complex mean and may reverse
    the signal's sign.

    The decay time ``tau`` has the same units as ``x``, and the frequency ``f``
    has the inverse units of ``x``. Note that the phase ``phi`` is in *turns*.
    """

    @classmethod
    @override
    def func(cls, x: ArrayLike, a, b, tau, f, phi) -> ArrayLike:
        return b + a * np.exp(-x / tau) * np.cos(2 * np.pi * (f * x + phi))

    @staticmethod
    def _has_uniform_steps(steps: np.ndarray) -> bool:
        """Check uniform spacing for a nonempty array of time steps."""
        return bool(np.allclose(steps, steps[0], rtol=1e-6, atol=0))

    @classmethod
    @override
    def guess(
        cls,
        preprocessed_data: xr.DataArray,
        coords: CurvefitCoordsType,
    ) -> CurvefitGuessType | None:
        """
        Crude initial guesses for damped oscillation parameters

        Supports real data with a named, one-dimensional, increasing,
        uniformly spaced numeric coordinate. Each trace must contain
        either only finite values or only NaN values.

        For finite, nonconstant traces, frequency is estimated using
        the FFT. The initial decay time is half the coordinate span.
        Amplitude, offset, and phase are estimated by linear least
        squares.

        Constant traces use zero amplitude and their constant value as
        the baseline. Their remaining initial values are numerical
        placeholders. The run method marks these traces as unsuccessful.

        All-NaN traces have NaN guesses for amplitude, offset, frequency,
        and phase. The provisional decay time is shared across traces
        and depends only on the coordinate.

        Returns None for unsupported coordinates, partially missing
        traces, or traces containing infinity.
        """
        y = preprocessed_data
        if not isinstance(coords, str):
            return None

        x = y[coords]
        if x.ndim != 1 or x.size < 3:
            return None

        dim = x.dims[0]

        time = x.to_numpy().astype(float)
        if not np.isfinite(time).all():
            return None

        finite_trace = np.isfinite(y).all(dim)
        all_nan_trace = y.isnull().all(dim)
        if not (finite_trace | all_nan_trace).all():
            return None

        steps = np.diff(time)
        if steps[0] <= 0 or not cls._has_uniform_steps(steps):
            # TODO: if we need it, consider adding guess for frequency even with
            # non-uniform step using e.g.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html
            return None

        # use non-default frequency_dim_name so that we don't conflict with a
        # possible dimension named 'frequency' on the data
        spec = simple_dft(y, coords, frequency_dim_name="_f")
        # Select only positive frequencies (because we're not using rfft) -
        # exclude the DC component too
        spec = spec.where(spec._f > 0, drop=True)

        # use argmax with skipna=False + isel instead of idxmax + sel, so that NaNs are handled correctly
        peak_idx = abs(spec).argmax("_f", skipna=False)
        # drop_vars so that the extra '_f' dimension is not in the result
        peak = spec.isel(_f=peak_idx).drop_vars("_f")
        peak_freq = spec._f.isel(_f=peak_idx).drop_vars("_f")

        # TODO: consider better guess for tau based on peak width
        tau = float((time[-1] - time[0]) / 2)
        baseline = y.mean(coords)
        # divide by envelope mean to account for the reduced amplitude due to the decay
        amplitude = 2 * abs(peak) / (x.size * np.exp(-x / tau).mean())
        phase = 2 * np.pi * np.arctan2(peak.imag, peak.real) # multiply by 2pi to get turns

        return {
            "a": amplitude,
            "b": baseline,
            "tau": tau,
            "f": peak_freq,
            "phi": phase,
        }

    @classmethod
    @override
    def run(
        cls,
        data: xr.DataArray,
        coords: CurvefitCoordsType,
        guess: CurvefitGuessType | None = None,
        curvefit_kwargs: dict[str, Any] | None = None,
    ) -> CurvefitAnalysisResult:
        """
        Fit damped oscillations with default decay-time bounds.

        The default bounds for ``tau`` are ``(0, np.inf)``.
        Explicitly supplied bounds override this default.

        The default optimization method is ``trf``. For ``trf`` and
        ``dogbox``, parameter scaling defaults to ``x_scale="jac"``.

        For a named one-dimensional coordinate, constant input traces are
        marked as unsuccessful. Their entries in ``params`` and
        ``fit_params`` are replaced by NaN.

        A named one-dimensional coordinate must contain only finite
        values, even when ``skipna=True`` is supplied.

        Named one-dimensional coordinates that are not strictly
        increasing or not uniformly spaced require explicit initial
        values for all five parameters. Fitting preserves the original
        sample order.

        Args:
            data: Data to analyze.
            coords: Coordinate(s) along which to perform curve fitting.
            guess: Initial parameter values overriding automatic guesses.
            curvefit_kwargs: Keyword arguments passed to Xarray curvefit.

        Returns:
            The curve-fitting analysis result.

        Raises:
            ValueError: If a named one-dimensional coordinate contains
                NaN or infinity, or if complete initial values are
                missing for non-increasing or nonuniform coordinates.
        """
        options = {} if curvefit_kwargs is None else dict(curvefit_kwargs)

        bounds = dict(options.get("bounds") or {})
        bounds.setdefault("tau", (0, np.inf))
        options["bounds"] = bounds

        scipy_kwargs = dict(options.get("kwargs") or {})
        if scipy_kwargs.get("method") is None:
            scipy_kwargs["method"] = "trf"
        if scipy_kwargs["method"] in ("trf", "dogbox"):
            scipy_kwargs.setdefault("x_scale", "jac")
        options["kwargs"] = scipy_kwargs

        constant = None
        if isinstance(coords, str):
            coordinate = data[coords]
            if coordinate.ndim == 1 and coordinate.size > 0:
                if not np.isfinite(coordinate).all():
                    raise ValueError(
                        "Time coordinates must contain only finite values."
                    )
                time = coordinate.to_numpy()
                manual_guess_reason = None

                if np.any(time[1:] <= time[:-1]):
                    manual_guess_reason = "Non-increasing"
                elif time.size >= 3:
                    steps = np.diff(time.astype(float))
                    if not cls._has_uniform_steps(steps):
                        manual_guess_reason = "Nonuniform"

                if manual_guess_reason is not None:
                    required = {"a", "b", "tau", "f", "phi"}
                    if guess is None or not required.issubset(guess):
                        raise ValueError(
                            f"{manual_guess_reason} time coordinates require initial "
                            "guesses for a, b, tau, f, and phi."
                        )

                dim = coordinate.dims[0]
                first = data.isel({dim: 0}, drop=True)
                constant = (data == first).all(dim)

        prepared_guess = {} if guess is None else dict(guess)

        if constant is not None and constant.any():
            preprocessed = cls.preprocess(data, coords=coords)
            data_to_guess = data if preprocessed is None else preprocessed
            automatic_guess = cls.guess(data_to_guess, coords=coords)

            if automatic_guess is not None:
                for name, (lower, upper) in bounds.items():
                    if name in prepared_guess or name not in automatic_guess:
                        continue

                    initial = automatic_guess[name]
                    bounded = np.minimum(np.maximum(initial, lower), upper)
                    prepared_guess[name] = xr.where(
                        constant,
                        bounded,
                        initial,
                    )

        result = super().run(
            data,
            coords=coords,
            guess=prepared_guess,
            curvefit_kwargs=options,
        )

        if constant is not None:
            result.success = result.success & ~constant
            result.params = result.params.where(~constant)
            result.fit_params = result.fit_params.where(~constant)

        return result

    @classmethod
    @override
    def preprocess(
        cls,
        data: xr.DataArray,
        coords: CurvefitCoordsType,
    ) -> xr.DataArray | None:
        """
        Project complex readout IQ to the real axis.

        Args:
            data: Real-valued data or complex readout IQ.
            coords: For complex input, the name of a one-dimensional
                coordinate along which to perform the projection.

        Returns:
            The projected data, or ``None`` for real-valued input.

        Raises:
            TypeError: If complex input uses a coordinate specification
                other than a string.
            ValueError: If the named coordinate is not one-dimensional.
        """
        if not np.iscomplexobj(data):
            return None

        if not isinstance(coords, str):
            raise TypeError("Complex data require a named one-dimensional coordinate.")

        coordinate = data[coords]
        if coordinate.ndim != 1:
            raise ValueError("Complex data require a named one-dimensional coordinate.")

        return project_complex(data, dim=coordinate.dims[0])


class ExponentialRegressionAnalysis(BaseAnalysis):
    r"""
    Analysis for exponentially decaying data with non-zero baseline

    Fits the model

    .. math::

        a \cdot \exp(-k \cdot x) + b

    to the data. Note that this is not a subclass of :py:class:`~sqe_analysis.analysis_base.CurvefitAnalysis`. The
    parameters are directly extracted from the data without any curve fitting
    or guessing, based on the method described by J. Jacquelin, see

    - https://stackoverflow.com/a/39436209
    - https://math.stackexchange.com/a/1337641
    - `Theoretical Impedances of Capacitive Electrodes <https://www.scribd.com/document/23155389/Theoretical-Impedance-of-Capacitive-Electrodes>`__
    - `Régressions et Équations Intégrales (in French) <https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales>`__

    Note that the parameter names are different from those used by Jacquelin.

    This method has the advantage that it is extremely fast and requires no
    initial guess. The downside is that it does not provide error bounds for the
    parameters. It can be used as an initial guess for curve fitting.

    This method also works for data that lies along a line in the complex plane,
    with complex-valued :math:`a` and :math:`b`. In this case, the decay
    parameter :math:`k` may also have a small imaginary component.
    """

    # TODO: example (with plot) showing that it also works for complex-valued data

    @classmethod
    @override
    def run(
        cls, data: xr.Dataset, dim: str | None = None, snr_threshold: float = 5.0
    ) -> CurvefitAnalysisResult:
        r"""
        Run the analysis.

        Args:
            data: The data to analyze
            dim: The dimension along which the exponential decay occurs, usually
                time. If ``None``, use the longest dimesnsion of the data.
            snr_threshold: Threshold for SNR below which the result is marked as
                a failure. If ``dim`` is :math:`t`, the SNR is calculated as

                .. math::

                    \mathrm{SNR} = |f(t=0) - f(t=t_{\mathrm{max}})| / |\mathrm{std}(\mathrm{data} - f(t))|,

                where :math:`f(t)` is the model function evaluated with the extracted
                parameters, and :math:`t_{\mathrm{max}}` is the maximum value of ``dim`` in the data.
                The standard deviation :math:`\mathrm{std}` is calculated over ``dim``.

        Returns:
            A :py:class:`~sqe_analysis.result.CurvefitAnalysisResult` (even though we are not doing curve
            fitting), with ``fit_params`` corresponding to the exponential
            parameters. ``params`` contains an additional parameter for the
            decay constant, which is the inverse of the scale factor :math:`k` in
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
                decay_constant=1 / fit_params.k,
                SNR=snr,
            ),
            analysis_class=cls,
            source_dataset_id=get_source_dataset_id(data),
            success=snr > snr_threshold,
            fit_params=fit_params,
        )

    @classmethod
    def func(cls, x: ArrayLike, k, a, b) -> ArrayLike:
        """
        Exponential model function.

        This can be used to conveniently evaluate the analysis result.
        """
        return a * np.exp(-k * x) + b


class GaussianAnalysis(CurvefitAnalysis):
    r"""
    Curve fit for Gaussian-shaped data

    Fits the model

    .. math::

        a \cdot \exp\left(-\frac{(x-c)^2}{2\sigma^2}\right) + b

    to the data.

    The data may be complex-valued, it will be projected to the real axis using
    :py:func:`~sqe_analysis.signal_processing.project_complex` in :py:meth:`preprocess`.
    """

    @classmethod
    @override
    def func(cls, x: ArrayLike, c, a, b, sigma) -> ArrayLike:
        return a * np.exp(-((x - c) ** 2) / (2 * sigma**2)) + b

    @classmethod
    @override
    def guess(
        cls,
        preprocessed_data: xr.DataArray,
        coords: CurvefitCoordsType,
    ) -> CurvefitGuessType:
        """
        Crude initial guess for Gaussian parameters
        """
        # TODO: consider factoring out in to "rough peak analysis" or similar

        y = preprocessed_data
        x = y[coords]
        mi = y.min(coords)
        ma = y.max(coords)

        # Use the maximum of the median absolute deviation as an estimate for
        # the center location. This is not very robust if the baseline is not
        # clearly visible. An alternative approach in this case would be to take
        # the integral of the signal, normalize it, and look for the point where
        # it crosses 0.5. But that method is not robust when there is a large
        # baseline visible but only part of the bell curve.
        center_loc = abs(y - y.median(coords)).idxmax(coords)

        # drop_vars so that the coordinate doesn't linger around
        center_val = y.sel({coords: center_loc}).drop_vars(coords)

        # if the estimated peak is close to the minimum, flip the sign
        sign = xr.where(abs(center_val - mi) < abs(center_val - ma), -1, 1)

        amplitude = (ma - mi) * sign
        baseline = xr.where(sign == -1, ma, mi)

        y_norm = (y - baseline) / amplitude

        above_half_max = x.where(y_norm > 0.5)
        above_half_max_range = above_half_max.max(coords) - above_half_max.min(coords)
        sigma = 0.5 * above_half_max_range

        return {
            "c": center_loc,
            "a": amplitude,
            "b": baseline,
            "sigma": sigma,
        }

    @classmethod
    @override
    def preprocess(cls, data: xr.DataArray, coords: str) -> xr.DataArray:
        """
        Project complex-valued data to real axis
        """
        proj = project_complex(data, dim=coords)
        return proj

    @classmethod
    @override
    def extra_params(cls, fit_params: xr.Dataset):
        r"""
        Add the following parameters to the fit result:

        - ``FWHM``: full width at half maximum, :math:`\sigma \times 2\sqrt{2 \ln 2}`
        """
        return xr.Dataset(
            {
                "FWHM": 2 * fit_params.sigma * np.sqrt(2 * np.log(2)),
            }
        )


class TimeOfFlightAnalysis(BaseAnalysis):
    """
    Analysis of time-of-flight measurement for calibrating acquisition delay

    Finds the location of a single step-like rising edge in the data. Assumes
    that the data is demodulated. The data may be complex-valued, it is
    projected to the real axis using
    :py:func:`~sqe_analysis.signal_processing.project_complex`. Always returns a
    result that is a value from the analysis axis (i.e. time) of the dataset.

    Currently, this method does not work reliably if there is both a rising and
    falling edge in the data.

    Example:

        >>> from sqe_analysis.analysis import TimeOfFlightAnalysis
        >>> from sqe_analysis.example_data import open_dataset
        >>> data = open_dataset("time_of_flight-good_snr_cut_off-RX4_30").Q60
        >>> result = TimeOfFlightAnalysis.run(data)
        >>> result.params.step_location.item()
        1144.0

    """

    # TODO: add visualization to the docstring

    @classmethod
    @override
    def run(
        cls,
        data: xr.DataArray,
        dim: Dims | None = None,
        snr_threshold: float = 2.0,
        smoothing: int = 5,
    ) -> AnalysisResult:
        """
        Perform the analysis.

        Args:
            dim: Dimension along which to perform the analysis, usually time. If
                ``None``, uses the longest dimension of the data.
            snr_threshold: If the SNR is below this value, the fit is marked as
                unsuccessful.
            smoothing: Size of rolling mean window along ``dim`` used to smooth
                the data before taking the derivative.

        Returns:
            An :py:class:`~sqe_analysis.result.AnalysisResult` with the
            following ``.params``:

            -  ``step_location`` - the location of the step
            - ``SNR`` - SNR estimated from the difference between the signal before and after the step

            The data projected to the real axis is stored in
            ``.intermediate_results.data_projected``, which may be used for
            visualization.
        """
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
            analysis_class=cls,
            source_dataset_id=get_source_dataset_id(data),
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
