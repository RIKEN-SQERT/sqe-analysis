"""Complex reflection and notch resonator analysis, independent of acquisition software."""

from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
from scipy.signal import peak_widths, savgol_filter

from sqe_analysis.analysis_base import (
    CurvefitAnalysis,
    CurvefitBoundsType,
    CurvefitGuessType,
)
from sqe_analysis.result import CurvefitAnalysisResult, get_source_dataset_id
from sqe_analysis.xarray_util import longest_dim

_PARAMS = ("fr", "Ql", "Qc", "amplitude", "phase", "delay", "phi")
_MODEL_PARAMS = (*_PARAMS, "reference_frequency")
_DERIVED = ("Qi", "linewidth")


class _ResonatorAnalysis(CurvefitAnalysis):
    """Shared complex least-squares backend; see the two public subclasses."""

    coupling_factor = 1

    @classmethod
    def func(
        cls, f, fr, Ql, Qc, amplitude, phase, delay, phi=0.0, reference_frequency=0.0
    ):
        r"""Evaluate the complex response, retaining Xarray broadcasting.

        ``Qc`` is the diameter-corrected, real external Q. The inverse complex
        coupling is ``(1 + 1j*tan(phi))/Qc``. Phase is in radians at
        reference_frequency, delay in seconds, and all frequencies in Hz.
        """
        background = amplitude * np.exp(
            1j * (phase - 2 * np.pi * (f - reference_frequency) * delay)
        )
        resonance = cls.coupling_factor * Ql / Qc * (1 + 1j * np.tan(phi))
        return background * (1 - resonance / (1 + 2j * Ql * (f - fr) / fr))

    @classmethod
    def bounds(cls):
        """Physical domains; per-trace frequency bounds are added by run()."""
        return {
            "Ql": (100.0, 1e10),
            "Qc": (100.0, 1e12),
            "amplitude": (1e-15, 1e15),
            "phi": (-1.2, 1.2),
        }

    @classmethod
    def extra_params(cls, fit_params):
        """Internal Q and loaded FWHM in Hz, using the real inverse coupling."""
        return xr.Dataset(
            {
                "Qi": 1 / (1 / fit_params.Ql - 1 / fit_params.Qc),
                "linewidth": fit_params.fr / fit_params.Ql,
            }
        )

    @classmethod
    def run(
        cls,
        data: xr.DataArray,
        coords: str | None = None,
        guess: CurvefitGuessType | None = None,
        bounds: CurvefitBoundsType | None = None,
        curvefit_kwargs: dict[str, Any] | None = None,
        *,
        max_relative_residual: float = 0.1,
        min_linewidth_points: float = 2.0,
        max_relative_std: float = 0.5,
    ) -> CurvefitAnalysisResult:
        """Fit complex linear-amplitude data along a 1-D frequency dimension.

        The calling and result conventions follow CurvefitAnalysis. Unlike its
        real-valued backend, this implementation minimizes stacked real and
        imaginary residuals with scipy.optimize.least_squares. Remaining
        dimensions are vectorized; no qubit, power or instrument names are
        assumed. ``coords=None`` selects the longest dimension.

        ``guess`` and ``bounds`` accept scalars or DataArrays on the remaining
        dimensions. Keys are fr, Ql, Qc, amplitude, phase, delay and phi.
        reference_frequency is fixed to the midpoint of each trace's valid
        frequency range and is included in fit_params for model evaluation.
        Positive Q/amplitude domains and ``abs(phi) < pi/2`` cannot be removed.

        curvefit_kwargs supports ``errors`` ('ignore' or 'raise'), ``skipna``
        (default True), and ``kwargs`` containing least_squares options
        max_nfev, ftol, xtol, gtol and verbose. Other Xarray curvefit options
        are not supported. Invalid configuration raises even with errors=ignore.
        Per-trace fit failures produce NaN parameters and success=False with
        errors=ignore (the default), or raise with errors=raise.

        Success additionally requires positive internal Q, an interior resolved
        resonance, identifiable covariance, and RMS residual below
        max_relative_residual times the fitted resonance-circle diameter.
        Ql and Qc relative standard errors must not exceed max_relative_std.
        The resonator model must also improve BIC by more than 10 relative
        to a constant-gain, linear-phase background (debug_results.delta_bic).
        A converged fit failing these quality checks retains its parameters
        but has success=False. Neither that flag nor covariance errors replace
        inspection for nonlinear response, drift, leakage or other resonances.
        Eager arrays are supported; compute Dask arrays before calling run().
        """
        if not isinstance(data, xr.DataArray) or not np.iscomplexobj(data):
            raise ValueError(
                "data must be a complex-valued Xarray DataArray (linear amplitude)"
            )
        dim = longest_dim(data) if coords is None else coords
        if not isinstance(dim, str) or dim not in data.dims or dim not in data.coords:
            raise ValueError(
                "coords must name a frequency dimension with a 1-D coordinate"
            )
        frequency = data[dim]
        if frequency.dims != (dim,) or not np.issubdtype(frequency.dtype, np.number):
            raise ValueError("frequency must be a numeric 1-D dimension coordinate")
        f = np.asarray(frequency, dtype=float)
        if not np.all(np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
            raise ValueError(
                "frequency must contain finite, positive, strictly increasing Hz values"
            )
        if len(f) < 8:
            raise ValueError("At least eight frequency samples are required")
        if not np.isfinite(max_relative_residual) or max_relative_residual <= 0:
            raise ValueError("max_relative_residual must be finite and positive")
        if not np.isfinite(min_linewidth_points) or min_linewidth_points <= 0:
            raise ValueError("min_linewidth_points must be finite and positive")
        if not np.isfinite(max_relative_std) or max_relative_std <= 0:
            raise ValueError("max_relative_std must be finite and positive")
        if data.chunks is not None:
            raise ValueError("Compute Dask arrays before fitting resonators")
        options = dict(curvefit_kwargs or {})
        errors = options.pop("errors", "ignore")
        skipna = options.pop("skipna", True)
        optimizer_options = dict(options.pop("kwargs", {}))
        if options or errors not in ("ignore", "raise") or not isinstance(skipna, bool):
            raise ValueError("Supported curvefit_kwargs: errors, skipna, kwargs")
        if set(optimizer_options) - {"max_nfev", "ftol", "xtol", "gtol", "verbose"}:
            raise ValueError("Unsupported least_squares option")
        optimizer_options = {"max_nfev": 3000, **optimizer_options}
        if (
            not isinstance(optimizer_options["max_nfev"], (int, np.integer))
            or optimizer_options["max_nfev"] <= 0
        ):
            raise ValueError("max_nfev must be a positive integer")
        if optimizer_options.get("verbose", 0) not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1 or 2")
        for key in ("ftol", "xtol", "gtol"):
            value = optimizer_options.get(key)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive and finite, or None")
        if all(
            key in optimizer_options and optimizer_options[key] is None
            for key in ("ftol", "xtol", "gtol")
        ):
            raise ValueError("At least one optimizer tolerance must be enabled")
        initial = dict(guess or {})
        effective_bounds = {**cls.bounds(), **(bounds or {})}
        if (set(initial) | set(effective_bounds)) - set(_PARAMS):
            raise ValueError(f"guess/bounds keys must be among {_PARAMS}")

        def value_array(value):
            array = (
                xr.DataArray(value) if not isinstance(value, xr.DataArray) else value
            )
            if set(array.dims) - (set(data.dims) - {dim}):
                raise ValueError(
                    "guess/bounds may only vary over non-frequency dimensions"
                )
            return array

        guesses = xr.concat(
            [value_array(initial.get(k, np.nan)) for k in _PARAMS],
            dim=xr.IndexVariable("_parameter", list(_PARAMS)),
        )
        lower = xr.concat(
            [
                value_array(effective_bounds.get(k, (-np.inf, np.inf))[0])
                for k in _PARAMS
            ],
            dim=xr.IndexVariable("_parameter", list(_PARAMS)),
        )
        upper = xr.concat(
            [
                value_array(effective_bounds.get(k, (-np.inf, np.inf))[1])
                for k in _PARAMS
            ],
            dim=xr.IndexVariable("_parameter", list(_PARAMS)),
        )
        # Configuration errors must never be silently converted to failed traces.
        if (
            bool((lower >= upper).any())
            or bool(np.isnan(lower).any())
            or bool(np.isnan(upper).any())
        ):
            raise ValueError("Every lower bound must be less than its upper bound")
        for k in ("Ql", "Qc", "amplitude"):
            if bool((lower.sel(_parameter=k) <= 0).any()):
                raise ValueError(f"{k} must have a strictly positive lower bound")
        if bool((lower.sel(_parameter="phi") <= -np.pi / 2).any()) or bool(
            (upper.sel(_parameter="phi") >= np.pi / 2).any()
        ):
            raise ValueError("phi bounds must lie strictly inside (-pi/2, pi/2)")
        for k, v in initial.items():
            a = value_array(v)
            if bool((~np.isfinite(a)).any()) or bool(
                ((a < lower.sel(_parameter=k)) | (a > upper.sel(_parameter=k))).any()
            ):
                raise ValueError(
                    f"Initial guess for {k} must be finite and inside bounds"
                )
        if bool((lower.sel(_parameter="fr") >= f[-1]).any()) or bool(
            (upper.sel(_parameter="fr") <= f[0]).any()
        ):
            raise ValueError("fr bounds must overlap the frequency coordinate")
        if "fr" in initial and bool(
            (
                (value_array(initial["fr"]) < f[0])
                | (value_array(initial["fr"]) > f[-1])
            ).any()
        ):
            raise ValueError("Initial fr must lie within the frequency coordinate")

        def fit_one(y, g, lo, hi):
            valid = np.isfinite(y.real) & np.isfinite(y.imag)
            ff, yy = f[valid], y[valid]
            empty = (
                np.full(8, np.nan),
                np.full(10, np.nan),
                np.full(8, np.nan),
                np.full((7, 7), np.nan),
                np.full(f.shape, complex(np.nan, np.nan)),
                np.nan,
                np.nan,
                False,
            )
            try:
                if (not skipna and not valid.all()) or len(ff) < 8:
                    raise ValueError("Insufficient finite complex samples")
                return cls._fit_trace(
                    ff,
                    yy,
                    f,
                    g,
                    lo,
                    hi,
                    optimizer_options,
                    max_relative_residual,
                    min_linewidth_points,
                    max_relative_std,
                )
            except (
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ):
                if errors == "raise":
                    raise
                return empty

        p, std, g, covariance, fitted, residual, delta_bic, success = xr.apply_ufunc(
            fit_one,
            data,
            guesses,
            lower,
            upper,
            input_core_dims=[[dim], ["_parameter"], ["_parameter"], ["_parameter"]],
            output_core_dims=[
                ["_model_parameter"],
                ["_quantity"],
                ["_model_parameter"],
                ["_cov_i", "_cov_j"],
                [dim],
                [],
                [],
                [],
            ],
            vectorize=True,
            join="exact",
        )
        params = p.assign_coords(_model_parameter=list(_MODEL_PARAMS)).to_dataset(
            "_model_parameter"
        )
        params_guess = g.assign_coords(_model_parameter=list(_MODEL_PARAMS)).to_dataset(
            "_model_parameter"
        )
        std = std.assign_coords(_quantity=[*_MODEL_PARAMS, *_DERIVED]).to_dataset(
            "_quantity"
        )
        extra = cls.extra_params(params)
        return CurvefitAnalysisResult(
            params=params.merge(extra),
            fit_params=params,
            params_std=std,
            success=success,
            fit_params_guess=params_guess,
            intermediate_results=xr.Dataset(
                {
                    "fitted_data": fitted.assign_coords({dim: frequency}),
                    "residual": data - fitted,
                    "relative_residual": residual,
                }
            ),
            debug_results=xr.Dataset(
                {
                    "fit_covariance": covariance.assign_coords(
                        _cov_i=list(_PARAMS), _cov_j=list(_PARAMS)
                    ),
                    "delta_bic": delta_bic,
                }
            ),
            analysis_class=cls,
            source_dataset_id=get_source_dataset_id(data),
        )

    @classmethod
    def _fit_trace(
        cls,
        f,
        z,
        full_f,
        user_guess,
        lower,
        upper,
        options,
        residual_limit,
        resolution,
        std_limit,
    ):
        origin, scale = (f[0] + f[-1]) / 2, np.ptp(f)
        x = (f - origin) / scale
        lower, upper = lower.copy(), upper.copy()
        lower[0] = max(lower[0], f[0])
        upper[0] = min(upper[0], f[-1])
        if lower[0] >= upper[0]:
            raise ValueError("fr bounds must overlap the measured frequency range")

        # Normalize frequency and delay, and optimize positive quantities in log space.
        def encode(p):
            v = p.copy()
            v[0] = (v[0] - origin) / scale
            v[1:4] = np.log(v[1:4])
            v[5] *= 2 * np.pi * scale
            return v

        def decode(v):
            p = v.copy()
            p[0] = origin + p[0] * scale
            p[1:4] = np.exp(p[1:4])
            p[5] /= 2 * np.pi * scale
            return p

        low, high = encode(lower), encode(upper)
        amplitude = abs(z)
        baseline = np.median(
            np.r_[amplitude[: max(2, len(f) // 5)], amplitude[-max(2, len(f) // 5) :]]
        )
        if baseline <= np.finfo(float).tiny:
            raise ValueError("No measurable complex response")
        smoothed = savgol_filter(amplitude**2, 7, 2)
        dip = int(np.argmin(smoothed))
        phase = np.unwrap(np.angle(z))
        phase_slope = np.gradient(savgol_filter(phase, 7, 2), f)
        phase_index = int(np.argmax(abs(phase_slope - np.median(phase_slope))))
        candidates = list(dict.fromkeys([dip, phase_index]))
        best = None
        best_initial = None
        for index in candidates:
            width = scale / 10
            if (
                0 < index < len(f) - 1
                and smoothed[index] < smoothed[index - 1]
                and smoothed[index] < smoothed[index + 1]
            ):
                width = max(
                    float(peak_widths(-smoothed, [index])[0][0])
                    * np.median(np.diff(f)),
                    2 * np.median(np.diff(f)),
                )
            for ratio in (0.08, 0.35, 0.75, 0.95):
                ql = np.clip(f[index] / width, lower[1] * 1.01, upper[1] / 1.01)
                p = np.array([f[index], ql, ql / ratio, 1.0, 0.0, 0.0, 0.0])
                core = cls.func(f, *p, reference_frequency=origin)
                gain = z / core
                slope, intercept = np.polyfit(x, np.unwrap(np.angle(gain)), 1)
                p[3:6] = [baseline, intercept, -slope / (2 * np.pi * scale)]
                p = np.clip(p, lower, upper)
                p = np.where(np.isfinite(user_guess), user_guess, p)
                if np.any(p < lower) or np.any(p > upper):
                    raise ValueError("Initial guess is outside the per-trace bounds")
                v0 = encode(p)

                def residual(v):
                    d = (
                        cls.func(f, *decode(v), reference_frequency=origin) - z
                    ) / baseline
                    return np.r_[d.real, d.imag]

                fit = least_squares(
                    residual, v0, bounds=(low, high), x_scale="jac", **options
                )
                if fit.success and (best is None or fit.cost < best.cost):
                    best, best_initial = fit, p.copy()
        if best is None:
            raise RuntimeError("Resonator fit did not converge")
        p = decode(best.x)
        prediction = cls.func(full_f, *p, reference_frequency=origin)
        # Transform the full covariance back to the public physical parameters.
        transform = np.diag([scale, p[1], p[2], p[3], 1, 1 / (2 * np.pi * scale), 1])
        identifiable = np.linalg.matrix_rank(best.jac) == 7
        cov = transform @ np.linalg.pinv(best.jac.T @ best.jac) @ transform.T
        cov *= 2 * best.cost / (2 * len(f) - 7)
        fr, ql, qc, amp, _, _, phi = p
        qi = 1 / (1 / ql - 1 / qc) if ql != qc else np.inf
        gradients = [
            np.array([0, qi**2 / ql**2, -(qi**2) / qc**2, 0, 0, 0, 0]),
            np.array([1 / ql, -fr / ql**2, 0, 0, 0, 0, 0]),
        ]
        standard = np.r_[
            np.sqrt(np.maximum(np.diag(cov), 0)),
            0.0,
            [np.sqrt(max(g @ cov @ g, 0)) for g in gradients],
        ]
        if not identifiable:
            standard[:] = np.nan
        diameter = cls.coupling_factor * amp * ql / qc / abs(np.cos(phi))
        relative_residual = float(
            np.sqrt(np.mean(abs(cls.func(f, *p, reference_frequency=origin) - z) ** 2))
            / diameter
        )
        # A smooth gain/delay baseline must not be mistaken for a resonance.
        slope, intercept = np.polyfit(x, phase, 1)

        def null_residual(v):
            d = (np.exp(v[0] + 1j * (v[1] + v[2] * x)) - z) / baseline
            return np.r_[d.real, d.imag]

        null_fit = least_squares(null_residual, [np.log(baseline), intercept, slope])
        tiny = np.finfo(float).tiny
        n = 2 * len(f)
        delta_bic = n * np.log(
            max(2 * null_fit.cost, tiny) / max(2 * best.cost, tiny)
        ) - 4 * np.log(n)
        interior = lower[0] < fr < upper[0]
        on_bound = np.any(abs(best.active_mask) > 0)
        success = bool(
            identifiable
            and not on_bound
            and interior
            and qc > ql > 0
            and fr / ql >= resolution * np.median(np.diff(f))
            and relative_residual <= residual_limit
            and standard[1] / ql <= std_limit
            and standard[2] / qc <= std_limit
            and delta_bic > 10
            and np.all(np.isfinite(standard))
        )
        return (
            np.r_[p, origin],
            standard,
            np.r_[best_initial, origin],
            cov,
            prediction,
            relative_residual,
            float(delta_bic),
            success,
        )


class ReflectionResonatorAnalysis(_ResonatorAnalysis):
    r"""Fit a one-port reflection resonance including complex gain and delay.

    The response is ``A*exp(i*(phase-2*pi*(f-f_ref)*delay))`` times
    ``1 - 2*(Ql/Qc)*(1+i*tan(phi))/(1+2i*Ql*(f-fr)/fr)``.
    The port topology, not the VNA trace label, determines the model: a
    reflection measured through a circulator may be recorded as S21.
    Input and output follow :meth:`run`; see the resonator user guide.
    """

    coupling_factor = 2


class NotchResonatorAnalysis(_ResonatorAnalysis):
    r"""Fit a hanger/notch transmission resonance including gain and delay.

    The response uses the same convention as ReflectionResonatorAnalysis,
    with coupling factor 1 instead of 2. Qc is real, diameter-corrected:
    ``1/Qi = 1/Ql - 1/Qc``; the magnitude of the complex coupling Q is
    ``Qc*cos(phi)``. This is not a fit of magnitude-only data and does not
    project complex data onto a real axis.
    """

    coupling_factor = 1
