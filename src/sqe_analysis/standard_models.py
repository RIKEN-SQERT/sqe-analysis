"""Real-valued peak, fringe, oscillation, and step-response analyses.

All public classes accept DataArrays and return CurvefitAnalysisResult. Frequencies
are cycles per coordinate unit and phases are turns, as in DampedOscillationAnalysis.
No acquisition-framework, hardware, lmfit, or experiment-project imports are used.
"""

from __future__ import annotations

import inspect

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks, peak_widths

from .analysis_base import CurvefitAnalysis
from .result import CurvefitAnalysisResult
from .signal_processing import detect_peaks, project_complex
from .xarray_util import longest_dim


def _vector_guess(data, coords, names, callback):
    def one(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 3:
            return tuple(1.0 for _ in names)
        return tuple(callback(x, y))

    outputs = xr.apply_ufunc(
        one,
        data[coords],
        data,
        input_core_dims=[[coords], [coords]],
        output_core_dims=[[] for _ in names],
        vectorize=True,
        output_dtypes=[float] * len(names),
    )
    return dict(zip(names, outputs))


def _peak_guess(x, y):
    edge = max(1, len(x) // 10)
    b = float(np.median(np.r_[y[:edge], y[-edge:]]))
    score = abs(y - b)
    index = int(score.argmax())
    width = np.ptp(x) / 8
    if 0 < index < len(x) - 1:
        indices, _ = find_peaks(score)
        if index in indices:
            width = float(peak_widths(score, [index])[0][0] * np.median(np.diff(x)))
    width = max(width, np.median(np.diff(x)))
    return float(y[index] - b), b, float(x[index]), width


def _double_guess(x, y, gaussian=False):
    """Fit a dominant peak, then seed the second from the largest residual."""
    a, b, c, width = _peak_guess(x, y)
    span, origin, scale = np.ptp(x), x[0], max(np.ptp(y), abs(a), 1e-30)
    u, v = (x - origin) / span, y / scale
    c0, w0 = (c - origin) / span, width / (2.355 if gaussian else 2) / span

    def model(t, amplitude, baseline, center, sigma):
        core = (
            np.exp(-0.5 * ((t - center) / sigma) ** 2)
            if gaussian
            else 1 / (1 + ((t - center) / sigma) ** 2)
        )
        return baseline + amplitude * core

    try:
        p, _ = curve_fit(
            model,
            u,
            v,
            p0=[a / scale, b / scale, c0, w0],
            bounds=([-np.inf, -np.inf, 0, 1e-6], [np.inf, np.inf, 1, 2]),
            max_nfev=3000,
            x_scale="jac",
        )
        residual = v - model(u, *p)
    except (ValueError, RuntimeError):
        p = np.array([a / scale, b / scale, c0, w0])
        residual = v - model(u, *p)
    # Reuse the public ranked detector; inspect both residual polarities.
    residual_data = xr.DataArray(residual, dims="sample", coords={"sample": x})
    candidates = []
    for dip in (False, True):
        detected = detect_peaks(residual_data, "sample", dip=dip, rank=3)
        candidates.extend(int(i) for i in detected["index"].values if i >= 0)
    candidates = [i for i in candidates if abs(u[i] - p[2]) > max(p[3] / 2, 2 / len(x))]
    second = (
        max(candidates, key=lambda i: abs(residual[i]))
        if candidates
        else int(np.argmax(abs(residual)))
    )
    centers = sorted([p[2], u[second]])
    if centers[1] - centers[0] < 1 / len(x):
        centers = [max(0.0, p[2] - p[3]), min(1.0, p[2] + p[3])]
    amplitudes = [np.interp(center, u, v) - p[1] for center in centers]
    return (
        amplitudes[0] * scale,
        origin + centers[0] * span,
        max(p[3] * span / 2, span / len(x)),
        amplitudes[1] * scale,
        origin + centers[1] * span,
        max(p[3] * span / 2, span / len(x)),
        p[1] * scale,
    )


def _oscillation_guess(x, y):
    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=0):
        uniform = np.linspace(x[0], x[-1], len(x))
        y, x = np.interp(uniform, x, y), uniform
        spacing = np.diff(x)
    spectrum = np.fft.rfft(y - y.mean())
    frequencies = np.fft.rfftfreq(len(x), spacing[0])
    index = 1 + int(np.argmax(abs(spectrum[1:])))
    f = frequencies[index]
    phi = np.angle(spectrum[index]) / (2 * np.pi) - f * x[0]
    return np.ptp(y) / 2, y.mean(), f, phi


class _RealTraceAnalysis(CurvefitAnalysis):
    @classmethod
    def preprocess(cls, data, coords):
        return project_complex(data, dim=coords) if np.iscomplexobj(data) else data

    @classmethod
    def run(
        cls,
        data: xr.DataArray,
        coords: str | None = None,
        guess=None,
        bounds=None,
        curvefit_kwargs=None,
    ) -> CurvefitAnalysisResult:
        coords = longest_dim(data) if coords is None else coords
        if not isinstance(coords, str) or coords not in data.dims:
            raise ValueError("These models require one named analysis dimension")
        x = np.asarray(data[coords])
        if (
            x.ndim != 1
            or x.size < 3
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError(
                "Analysis coordinates must be finite and strictly increasing"
            )
        # Relative parameter scales differ strongly for GHz frequencies and ns times.
        options = dict(curvefit_kwargs or {})
        names = list(inspect.signature(cls.func).parameters)[1:]
        span = np.ptp(x)

        def jacobian(values, *parameters):
            columns = []
            for i, (name, value) in enumerate(zip(names, parameters)):
                characteristic = (
                    span
                    if name in {"c", "c1", "c2", "sigma", "sigma1", "sigma2", "tau"}
                    else 1 / span
                    if name in {"f", "s"}
                    else 1.0
                )
                h = np.sqrt(np.finfo(float).eps) * max(abs(value), characteristic)
                if name in {"sigma", "sigma1", "sigma2", "tau"} and value > 0:
                    h = min(h, value / 4)
                plus, minus = list(parameters), list(parameters)
                plus[i], minus[i] = value + h, value - h
                columns.append(
                    (cls.func(values, *plus) - cls.func(values, *minus)) / (2 * h)
                )
            return np.column_stack(columns)

        custom_solver = dict(options.pop("kwargs", {}))
        method = custom_solver.get("method", "trf")
        solver = {"method": method, "jac": jacobian}
        if method == "lm":
            solver["maxfev"] = 10000
        else:
            solver.update(x_scale="jac", max_nfev=custom_solver.pop("maxfev", 10000))
        solver.update(custom_solver)
        options["kwargs"] = solver
        result = super().run(data, coords, guess, bounds, options)
        used = cls.preprocess(data, coords)
        fitted = cls.func(data[coords], **result.fit_params)
        intermediate = result.intermediate_results or xr.Dataset()
        result.intermediate_results = intermediate.assign(
            fitted_data=fitted, residual=used - fitted
        )
        return result


class LinearAnalysis(_RealTraceAnalysis):
    """Fit ``a*x+b``. Parameters a and b are slope and intercept."""

    @classmethod
    def func(cls, x, a, b):
        return a * x + b

    @classmethod
    def guess(cls, preprocessed_data, coords):
        return _vector_guess(
            preprocessed_data, coords, ["a", "b"], lambda x, y: np.polyfit(x, y, 1)
        )


class LorentzianAnalysis(_RealTraceAnalysis):
    """Fit ``b+a/(1+((x-c)/sigma)**2)``; a is signed peak height, sigma is HWHM."""

    @classmethod
    def func(cls, x, a, b, c, sigma):
        return b + a / (1 + ((x - c) / sigma) ** 2)

    @classmethod
    def guess(cls, preprocessed_data, coords):
        def one(x, y):
            a, b, c, width = _peak_guess(x, y)
            return a, b, c, width / 2

        return _vector_guess(preprocessed_data, coords, ["a", "b", "c", "sigma"], one)

    @classmethod
    def bounds(cls):
        return {"sigma": (np.finfo(float).tiny, np.inf)}

    @classmethod
    def extra_params(cls, fit_params):
        return xr.Dataset({"FWHM": 2 * fit_params.sigma})

    @classmethod
    def extra_params_std(cls, fit_params, covariance):
        return xr.Dataset(
            {
                "FWHM": 2
                * np.sqrt(covariance.sel(cov_i="sigma", cov_j="sigma", drop=True))
            }
        )


class DoubleGaussianAnalysis(_RealTraceAnalysis):
    """Fit two signed Gaussian peaks plus one constant baseline.

    Parameters a1/c1/sigma1 and a2/c2/sigma2 describe height/center/standard width.
    Component labels follow the seeds and are not guaranteed to remain ordered.
    """

    @classmethod
    def func(cls, x, a1, c1, sigma1, a2, c2, sigma2, b):
        return (
            b
            + a1 * np.exp(-0.5 * ((x - c1) / sigma1) ** 2)
            + a2 * np.exp(-0.5 * ((x - c2) / sigma2) ** 2)
        )

    @classmethod
    def guess(cls, preprocessed_data, coords):
        return _vector_guess(
            preprocessed_data,
            coords,
            ["a1", "c1", "sigma1", "a2", "c2", "sigma2", "b"],
            lambda x, y: _double_guess(x, y, gaussian=True),
        )

    @classmethod
    def bounds(cls):
        return {
            "sigma1": (np.finfo(float).tiny, np.inf),
            "sigma2": (np.finfo(float).tiny, np.inf),
        }

    @classmethod
    def extra_params(cls, fit_params):
        return xr.Dataset(
            {
                f"FWHM{i}": 2 * np.sqrt(2 * np.log(2)) * fit_params[f"sigma{i}"]
                for i in (1, 2)
            }
        )

    @classmethod
    def extra_params_std(cls, fit_params, covariance):
        return xr.Dataset(
            {
                f"FWHM{i}": 2
                * np.sqrt(2 * np.log(2))
                * np.sqrt(
                    covariance.sel(cov_i=f"sigma{i}", cov_j=f"sigma{i}", drop=True)
                )
                for i in (1, 2)
            }
        )


class DoubleLorentzianAnalysis(DoubleGaussianAnalysis):
    """Fit two signed Lorentzians plus baseline; seed the second from a single-fit residual.

    Each sigma is HWHM. Component amplitudes are heights, not integrated areas.
    """

    @classmethod
    def func(cls, x, a1, c1, sigma1, a2, c2, sigma2, b):
        return (
            b
            + a1 / (1 + ((x - c1) / sigma1) ** 2)
            + a2 / (1 + ((x - c2) / sigma2) ** 2)
        )

    @classmethod
    def guess(cls, preprocessed_data, coords):
        return _vector_guess(
            preprocessed_data,
            coords,
            ["a1", "c1", "sigma1", "a2", "c2", "sigma2", "b"],
            _double_guess,
        )

    @classmethod
    def extra_params(cls, fit_params):
        return xr.Dataset({f"FWHM{i}": 2 * fit_params[f"sigma{i}"] for i in (1, 2)})

    @classmethod
    def extra_params_std(cls, fit_params, covariance):
        return xr.Dataset(
            {
                f"FWHM{i}": 2
                * np.sqrt(
                    covariance.sel(cov_i=f"sigma{i}", cov_j=f"sigma{i}", drop=True)
                )
                for i in (1, 2)
            }
        )


class OscillationAnalysis(_RealTraceAnalysis):
    """Fit ``b+a*cos(2*pi*(f*x+phi))`` without an artificial decay parameter.

    phi is in turns; period/half_period/quarter_period are intervals, not
    phase-corrected pulse amplitudes or certified pi-pulse calibrations.
    """

    @classmethod
    def func(cls, x, a, b, f, phi):
        return b + a * np.cos(2 * np.pi * (f * x + phi))

    @classmethod
    def guess(cls, preprocessed_data, coords):
        return _vector_guess(
            preprocessed_data, coords, ["a", "b", "f", "phi"], _oscillation_guess
        )

    @classmethod
    def bounds(cls):
        return {"f": (np.finfo(float).tiny, np.inf)}

    @classmethod
    def extra_params(cls, fit_params):
        return xr.Dataset(
            {
                name: factor / fit_params.f
                for name, factor in [
                    ("period", 1),
                    ("half_period", 0.5),
                    ("quarter_period", 0.25),
                ]
            }
        )

    @classmethod
    def extra_params_std(cls, fit_params, covariance):
        error = (
            np.sqrt(covariance.sel(cov_i="f", cov_j="f", drop=True)) / fit_params.f**2
        )
        return xr.Dataset(
            {
                name: factor * error
                for name, factor in [
                    ("period", 1),
                    ("half_period", 0.5),
                    ("quarter_period", 0.25),
                ]
            }
        )


class DampedBeatingOscillationAnalysis(_RealTraceAnalysis):
    """Fit ``b+a*exp(-x/tau)*(cos(2*pi*f*(x-c))+cos(2*pi*(f+s)*(x-c)))``.

    Both tones share amplitude, time shift c, and decay time, matching a simple
    beating model. f and s are positive; second_frequency=f+s. Automatic guesses
    require at least two resolvable spectral peaks; provide manual guesses otherwise.
    """

    @classmethod
    def func(cls, x, a, b, tau, f, s, c):
        return b + a * np.exp(-x / tau) * (
            np.cos(2 * np.pi * f * (x - c)) + np.cos(2 * np.pi * (f + s) * (x - c))
        )

    @classmethod
    def guess(cls, preprocessed_data, coords):
        def one(x, y):
            uniform = np.linspace(x[0], x[-1], len(x))
            y, x = np.interp(uniform, x, y), uniform
            a, b, f, phi = _oscillation_guess(x, y)
            spec = abs(np.fft.rfft(y - b))
            fs = np.fft.rfftfreq(len(x), x[1] - x[0])
            indices, _ = find_peaks(spec)
            indices = indices[indices > 0]
            if len(indices) >= 2:
                chosen = indices[np.argsort(spec[indices])[-2:]]
                f1, f2 = sorted(fs[chosen])
            else:
                f1, f2 = f, f + 1 / np.ptp(x)
            return a / 2, b, np.ptp(x) / 2, f1, f2 - f1, -phi / f

        return _vector_guess(
            preprocessed_data, coords, ["a", "b", "tau", "f", "s", "c"], one
        )

    @classmethod
    def bounds(cls):
        return {key: (np.finfo(float).tiny, np.inf) for key in ("tau", "f", "s")}

    @classmethod
    def extra_params(cls, fit_params):
        return xr.Dataset({"second_frequency": fit_params.f + fit_params.s})

    @classmethod
    def extra_params_std(cls, fit_params, covariance):
        variance = (
            covariance.sel(cov_i="f", cov_j="f", drop=True)
            + covariance.sel(cov_i="s", cov_j="s", drop=True)
            + 2 * covariance.sel(cov_i="f", cov_j="s", drop=True)
        )
        return xr.Dataset({"second_frequency": np.sqrt(variance.where(variance >= 0))})


class RamseyFringeAnalysis(_RealTraceAnalysis):
    """Gaussian-windowed fringe: ``b+a*(1-contrast*cos(2*pi*(f*(x-c)+phi)))*G``.

    G=exp(-0.5*((x-c)/sigma)**2). x can be a swept drive frequency; f is then
    its inverse, not the qubit frequency. c is the envelope center. The derived
    extremum_location is the global signed extremum within the measured range;
    it is not assumed equal to c. Its uncertainty is deliberately unavailable.
    """

    @classmethod
    def func(cls, x, a, b, c, sigma, f, phi, contrast):
        return b + a * (
            1 - contrast * np.cos(2 * np.pi * (f * (x - c) + phi))
        ) * np.exp(-0.5 * ((x - c) / sigma) ** 2)

    @classmethod
    def guess(cls, preprocessed_data, coords):
        def one(x, y):
            uniform = np.linspace(x[0], x[-1], len(x))
            y, x = np.interp(uniform, x, y), uniform
            a, b, c, _width = _peak_guess(x, y)
            weights = abs(y - b)
            if weights.sum() > 0:
                c = np.sum(x * weights) / weights.sum()
                sigma = np.sqrt(np.sum((x - c) ** 2 * weights) / weights.sum())
            else:
                sigma = np.ptp(x) / 4
            spectrum = np.fft.rfft(y - b)
            fs = np.fft.rfftfreq(len(x), x[1] - x[0])
            candidates, _ = find_peaks(abs(spectrum))
            candidates = candidates[fs[candidates] >= 2 / np.ptp(x)]
            index = (
                max(candidates, key=lambda i: abs(spectrum[i]))
                if len(candidates)
                else 1
            )
            f = fs[index]
            phi = (
                np.angle(spectrum[index]) / (2 * np.pi)
                + f * (c - x[0])
                - (0.5 if a > 0 else 0)
            ) % 1
            return a / 1.7, b, c, max(sigma, np.median(np.diff(x))), f, phi, 0.7

        return _vector_guess(
            preprocessed_data,
            coords,
            ["a", "b", "c", "sigma", "f", "phi", "contrast"],
            one,
        )

    @classmethod
    def bounds(cls):
        return {
            "sigma": (np.finfo(float).tiny, np.inf),
            "f": (np.finfo(float).tiny, np.inf),
            "contrast": (0.0, 1.0),
        }

    @classmethod
    def extra_params(cls, fit_params):
        return xr.Dataset(
            {
                "FWHM": 2 * np.sqrt(2 * np.log(2)) * fit_params.sigma,
                "fringe_period": 1 / fit_params.f,
            }
        )

    @classmethod
    def extra_params_std(cls, fit_params, covariance):
        return xr.Dataset(
            {
                "FWHM": 2
                * np.sqrt(2 * np.log(2))
                * np.sqrt(covariance.sel(cov_i="sigma", cov_j="sigma", drop=True)),
                "fringe_period": np.sqrt(
                    covariance.sel(cov_i="f", cov_j="f", drop=True)
                )
                / fit_params.f**2,
            }
        )

    @classmethod
    def run(cls, data, coords=None, guess=None, bounds=None, curvefit_kwargs=None):
        coords = longest_dim(data) if coords is None else coords
        result = super().run(data, coords, guess, bounds, curvefit_kwargs)
        names = list(inspect.signature(cls.func).parameters)[1:]
        low, high = float(data[coords][0]), float(data[coords][-1])

        def extremum(*parameters):
            if not np.all(np.isfinite(parameters)):
                return np.nan
            p = dict(zip(names, parameters))
            grid = np.linspace(
                low, high, min(100001, max(2001, int(abs(p["f"]) * (high - low) * 100)))
            )
            values = cls.func(grid, **p) * np.sign(p["a"])
            i = int(np.argmax(values))
            if i in (0, len(grid) - 1):
                return grid[i]
            center = grid[i]
            opt = minimize_scalar(
                lambda dx: -cls.func(center + dx, **p) * np.sign(p["a"]),
                bounds=(grid[i - 1] - center, grid[i + 1] - center),
                method="bounded",
                options={"xatol": max(np.ptp(grid) * 1e-12, np.finfo(float).eps)},
            )
            return center + opt.x

        location = xr.apply_ufunc(
            extremum,
            *(result.fit_params[n] for n in names),
            vectorize=True,
            output_dtypes=[float],
        )
        result.params["extremum_location"] = location
        result.params_std["extremum_location"] = xr.full_like(location, np.nan)
        return result
