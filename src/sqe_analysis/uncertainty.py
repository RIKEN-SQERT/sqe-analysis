"""Covariance utilities for fitted models and the integral exponential estimator."""

import numpy as np
import xarray as xr


def parameter_stds(covariance: xr.DataArray) -> xr.Dataset:
    """Return marginal standard deviations; unavailable variances remain NaN."""
    return xr.Dataset(
        {
            str(name): np.sqrt(
                covariance.sel(cov_i=name, cov_j=name, drop=True).where(
                    lambda v: v >= 0
                )
            )
            for name in covariance.cov_i.values
        }
    )


def regression_covariance(x, y):
    """Delta-method covariance of the actual integral-regression estimator.

    Assumes real samples with independent, equal-variance additive errors. The
    estimator Jacobian is analytic, including the dependence of its exponential
    basis on the observations. Complex/no-signal/missing-data cases return NaN.
    Parameter order is k, a, b. No least-squares refinement changes the estimate.
    """
    empty = np.full((3, 3), np.nan)
    if np.iscomplexobj(y) or len(y) <= 3 or not np.all(np.isfinite(y)):
        return empty
    span = np.ptp(x)
    if not np.isfinite(span) or span <= 0 or np.any(np.diff(x) <= 0):
        return empty
    x = np.asarray(x, float) / span
    y = np.asarray(y, float)
    dt = np.diff(x)
    s = np.r_[0.0, np.cumsum(dt * (y[1:] + y[:-1]) / 2)]

    def transpose_integral(v):
        tail = np.cumsum(v[::-1])[::-1]
        out = np.zeros_like(v)
        out[:-1] += dt / 2 * tail[1:]
        out[1:] += dt / 2 * tail[1:]
        return out

    u, v = x - x[0], y - y[0]
    m00, m01, m11 = u @ u, u @ s, s @ s
    v0, v1 = v @ u, v @ s
    det = m00 * m11 - m01**2
    if det <= np.finfo(float).eps * max(abs(m00 * m11), 1e-300):
        return empty
    c = (-m01 * v0 + m00 * v1) / det
    g01, g11 = transpose_integral(u), 2 * transpose_integral(s)
    gv0, gv1 = u.copy(), s + transpose_integral(v)
    gv0[0] -= u.sum()
    gv1[0] -= s.sum()
    gc = (-v0 * g01 - m01 * gv0 + m00 * gv1 - c * (m00 * g11 - 2 * m01 * g01)) / det
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(c * x)
        n, se, see, sy, sye = len(x), e.sum(), e @ e, y.sum(), y @ e
        d = n * see - se**2
        if d <= np.finfo(float).eps * max(abs(n * see), 1e-300):
            return empty
        a, b = (n * sye - se * sy) / d, (see * sy - se * sye) / d
        ge = np.sum(x * e) * gc
        gee = 2 * np.sum(x * e**2) * gc
        gye = e + np.sum(y * x * e) * gc
        gd = n * gee - 2 * se * ge
        ga = (n * gye - sy * ge - se - a * gd) / d
        gb = (sy * gee + see - sye * ge - se * gye - b * gd) / d
        jac = np.array([-gc / span, ga, gb])
        variance = np.sum((y - a * e - b) ** 2) / (n - 3)
        covariance = variance * (jac @ jac.T)
    return covariance if np.all(np.isfinite(covariance)) else empty
