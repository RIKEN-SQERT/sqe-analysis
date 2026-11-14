"""Discrete edge profiling followed by a conditional exponential fit."""

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from .analysis_base import CurvefitAnalysis
from .result import CurvefitAnalysisResult, get_source_dataset_id
from .uncertainty import parameter_stds
from .xarray_util import longest_dim


class StepExponentialAnalysis(CurvefitAnalysis):
    """Fit H(x-c)*(a+b*exp(-(x-c)/tau)) to a real step/decay response.

    c is a discrete edge on the observed sample grid, not a differentiable fit
    parameter. The largest sample jumps seed a profile over candidate edges;
    a/b/tau are fitted at each candidate and the smallest SSE is retained.
    ``params_std`` contains conditional covariance errors for a/b/tau and NaN
    for c. debug_results.edge_lower/edge_upper bracket the unresolved edge.
    This avoids misleading zero uncertainty from differentiating a hard step.
    The pre-edge signal is zero; subtract an independently known baseline first.
    """

    @classmethod
    def func(cls, x, a, b, tau, c):
        dt = np.maximum(x - c, 0)
        return (x >= c) * (a + b * np.exp(-dt / tau))

    @classmethod
    def run(
        cls,
        data,
        coords=None,
        guess=None,
        bounds=None,
        curvefit_kwargs=None,
        *,
        edge_candidates=8,
    ):
        coords = longest_dim(data) if coords is None else coords
        if not isinstance(coords, str) or coords not in data.dims:
            raise ValueError("A named analysis dimension is required")
        x = np.asarray(data[coords], float)
        if len(x) < 6 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError(
                "At least six finite, increasing sample coordinates are required"
            )
        if (
            not isinstance(edge_candidates, int)
            or isinstance(edge_candidates, bool)
            or edge_candidates < 1
        ):
            raise ValueError("edge_candidates must be a positive integer")
        names = ["a", "b", "tau", "c"]
        guess, bounds = dict(guess or {}), dict(bounds or {})
        if set(guess) - set(names) or set(bounds) - set(names):
            raise ValueError("Unknown parameter name")
        options = dict(curvefit_kwargs or {})
        if set(options) - {"errors", "skipna", "kwargs"}:
            raise ValueError("Supported curvefit_kwargs: errors, skipna, kwargs")
        errors, skipna = options.get("errors", "ignore"), options.get("skipna", True)
        if errors not in {"ignore", "raise"}:
            raise ValueError("errors must be ignore or raise")
        solver = dict(options.get("kwargs", {}))
        if set(solver) - {"max_nfev", "ftol", "xtol", "gtol"}:
            raise ValueError("Unsupported step-fit solver option")
        if np.iscomplexobj(data):
            raise ValueError(
                "Step fitting requires real input with a zero pre-edge baseline"
            )
        used = data
        span = np.ptp(x)
        defaults = {
            "a": (-np.inf, np.inf),
            "b": (-np.inf, np.inf),
            "tau": (np.finfo(float).tiny, np.inf),
            "c": (x[0], x[-1]),
        }
        effective = {**defaults, **bounds}
        args = [
            xr.DataArray(guess.get(n, np.nan))
            if not isinstance(guess.get(n), xr.DataArray)
            else guess[n]
            for n in names
        ]
        limits = []
        for name in names:
            low, high = effective[name]
            if np.any(np.asarray(low) >= np.asarray(high)):
                raise ValueError("Each lower bound must be below its upper bound")
            limits.extend(
                [
                    low if isinstance(low, xr.DataArray) else xr.DataArray(low),
                    high if isinstance(high, xr.DataArray) else xr.DataArray(high),
                ]
            )

        def one(y, *inputs):
            g, lower, upper = (
                np.array(inputs[:4], float),
                np.array(inputs[4::2]),
                np.array(inputs[5::2]),
            )
            if any(
                name in guess and not np.isfinite(g[i]) for i, name in enumerate(names)
            ):
                raise ValueError("Manual guesses must be finite")
            for i, name in enumerate(names):
                if name in guess and not lower[i] <= g[i] <= upper[i]:
                    raise ValueError("Initial guess is outside effective bounds")
            valid = np.isfinite(y)
            if not skipna and not valid.all():
                if errors == "raise":
                    raise ValueError("Nonfinite data with skipna=False")
                return (
                    np.full(4, np.nan),
                    np.full((4, 4), np.nan),
                    np.nan,
                    np.nan,
                    False,
                )
            xx, yy = x[valid], y[valid]
            empty = (np.full(4, np.nan), np.full((4, 4), np.nan), np.nan, np.nan, False)
            if len(xx) < 6:
                if errors == "raise":
                    raise ValueError("Insufficient finite samples")
                return empty
            jumps = abs(np.diff(yy))
            candidates = list(1 + np.argsort(-jumps)[:edge_candidates])
            if "c" in guess:
                index = int(np.searchsorted(xx, g[3]))
                candidates.extend(range(max(1, index - 2), min(len(xx) - 3, index + 3)))
            scale = max(np.ptp(yy), np.max(abs(yy)), 1e-30)
            transform = np.array([scale, scale, span])
            lo, hi = lower[:3] / transform, upper[:3] / transform
            best = None
            for index in sorted(set(candidates)):
                c = xx[index]
                if not lower[3] <= c <= upper[3] or len(xx) - index < 4:
                    continue
                a = np.median(yy[-max(1, (len(xx) - index) // 5) :])
                initial = np.array([a, yy[index] - a, (xx[-1] - c) / 3])
                for i, name in enumerate(names[:3]):
                    if name in guess:
                        initial[i] = g[i]
                p0 = initial / transform
                if np.any(p0 < lo) or np.any(p0 > hi):
                    # Automatic estimates may be clipped; manual out-of-bounds values were rejected above.
                    p0 = np.clip(
                        p0,
                        lo + np.where(np.isfinite(lo), 1e-12, 0),
                        hi - np.where(np.isfinite(hi), 1e-12, 0),
                    )
                t = (xx - c) / span

                def model(t, a, b, tau):
                    return (t >= 0) * (a + b * np.exp(-np.maximum(t, 0) / tau))

                try:
                    p, cov = curve_fit(
                        model,
                        t,
                        yy / scale,
                        p0=p0,
                        bounds=(lo, hi),
                        method="trf",
                        x_scale="jac",
                        **{"max_nfev": 5000, **solver},
                    )
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
                sse = float(np.sum((yy / scale - model(t, *p)) ** 2))
                if best is None or sse < best[0]:
                    covariance = np.full((4, 4), np.nan)
                    covariance[:3, :3] = cov * transform[:, None] * transform[None, :]
                    best = (sse, np.r_[p * transform, c], covariance, xx[index - 1], c)
            if best is None:
                if errors == "raise":
                    raise RuntimeError("No edge candidate yielded a valid fit")
                return empty
            return *best[1:], True

        values, covariance, edge_lower, edge_upper, success = xr.apply_ufunc(
            one,
            used,
            *args,
            *limits,
            input_core_dims=[[coords]] + [[]] * 12,
            output_core_dims=[["param"], ["cov_i", "cov_j"], [], [], []],
            vectorize=True,
            output_dtypes=[float, float, float, float, bool],
        )
        values = values.assign_coords(param=names)
        covariance = covariance.assign_coords(cov_i=names, cov_j=names)
        params = values.to_dataset("param")
        fitted = cls.func(data[coords], **params)
        return CurvefitAnalysisResult(
            params=params,
            fit_params=params,
            params_std=parameter_stds(covariance),
            success=success,
            analysis_class=cls,
            source_dataset_id=get_source_dataset_id(data),
            fit_params_guess=xr.Dataset(guess) if guess else None,
            intermediate_results=xr.Dataset(
                {
                    "preprocessed_data": used,
                    "fitted_data": fitted,
                    "residual": used - fitted,
                }
            ),
            debug_results=xr.Dataset(
                {
                    "covariance": covariance,
                    "edge_lower": edge_lower,
                    "edge_upper": edge_upper,
                }
            ),
        )
