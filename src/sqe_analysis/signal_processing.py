"""
Functions for digital signal processing with Xarray

These should complement functions available in the base Xarray library and in
other external libraries such as `xr-scipy <https://xr-scipy.readthedocs.io>`_
or `xrft <https://xrft.readthedocs.io/>`_. Those libraries should be preferred
whenever the desired functionality is available.
"""

from typing import Literal, cast, overload

import numpy as np
import xarray as xr
from xarray.core.types import Dims

from sqe_analysis.xarray_util import longest_dim


def detect_peaks(
    data: xr.DataArray,
    dim: str | None = None,
    *,
    dip: bool | None = None,
    rank: int = 2,
    prominence: float | None = None,
    distance: int = 1,
    peak_dim: str = "peak",
) -> xr.Dataset:
    """Find local peaks/dips and rank by strength, independently along each trace.

    Input must be real with a finite, increasing coordinate. ``dip=None`` chooses
    the sign of the largest deviation from the median for each trace. ``rank``
    fixes the result length: missing positions/heights/strengths are NaN and
    missing indices are -1. ``count`` counts all detected peaks, not only retained
    ones. ``peak_mask`` marks all candidates on the original sample grid.
    Heights are original values; strengths are measured from the trace minimum
    (peaks) or maximum (dips). Endpoints and peaks adjacent to NaNs are excluded.
    Prominence is in signal units; distance is a positive integer in samples.
    No interpolation or peak-position standard deviation is inferred.
    """
    from scipy.signal import find_peaks

    dim = longest_dim(data) if dim is None else dim
    if not isinstance(rank, (int, np.integer)) or isinstance(rank, bool) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if (
        not isinstance(distance, (int, np.integer))
        or isinstance(distance, bool)
        or distance < 1
    ):
        raise ValueError("distance must be a positive integer")
    if dip is not None and not isinstance(dip, bool):
        raise ValueError("dip must be True, False, or None")
    if prominence is not None and (not np.isfinite(prominence) or prominence < 0):
        raise ValueError("prominence must be finite and nonnegative")
    if peak_dim in data.dims or peak_dim in data.coords:
        raise ValueError(
            "peak_dim must not collide with an existing dimension/coordinate"
        )
    x = np.asarray(data[dim])
    if (
        x.ndim != 1
        or x.size < 3
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError(
            "At least three finite, increasing coordinate values are required"
        )
    if np.iscomplexobj(data):
        raise ValueError("Project complex data to real values before peak detection")

    def one(y):
        finite = np.isfinite(y)
        positions, heights, strengths = (np.full(rank, np.nan) for _ in range(3))
        indices, mask = np.full(rank, -1, dtype=int), np.zeros(y.size, dtype=bool)
        if not finite.any():
            return positions, heights, strengths, indices, 0, mask, False
        is_dip = (
            dip
            if dip is not None
            else bool(
                y[finite][np.argmax(abs(y[finite] - np.median(y[finite])))]
                < np.median(y[finite])
            )
        )
        score = np.where(
            finite, np.max(y[finite]) - y if is_dip else y - np.min(y[finite]), -np.inf
        )
        found, _ = find_peaks(score, prominence=prominence, distance=distance)
        found = found[finite[found - 1] & finite[found + 1]]
        mask[found] = True
        selected = found[np.argsort(-score[found], kind="stable")[:rank]]
        n = len(selected)
        positions[:n], heights[:n], strengths[:n], indices[:n] = (
            x[selected],
            y[selected],
            score[selected],
            selected,
        )
        return positions, heights, strengths, indices, len(found), mask, is_dip

    results = xr.apply_ufunc(
        one,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[
            [peak_dim],
            [peak_dim],
            [peak_dim],
            [peak_dim],
            [],
            [dim],
            [],
        ],
        vectorize=True,
        output_dtypes=[float, float, float, int, int, bool, bool],
    )
    output = xr.Dataset(
        dict(
            zip(
                [
                    "position",
                    "height",
                    "strength",
                    "index",
                    "count",
                    "peak_mask",
                    "is_dip",
                ],
                results,
            )
        )
    )
    return output.assign_coords(
        {peak_dim: np.arange(rank), dim: data[dim]}
    ).assign_attrs(data.attrs)


@overload
def project_complex(
    data: xr.DataArray, discard_imag: bool = True, dim: Dims | None = None
) -> xr.DataArray: ...


@overload
def project_complex(
    data: xr.Dataset, discard_imag: bool = True, dim: Dims | None = None
) -> xr.Dataset: ...


def project_complex(
    data: xr.DataArray | xr.Dataset,
    discard_imag: bool = True,
    dim: Dims | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    A simple method to project complex-valued data to the real axis in a way that maximizes the signal

    Note that the sign of the signal is not guaranteed. The same input data
    rotated slightly in the complex plane can result in output with the opposite
    sign.
    """

    if dim is None:
        dim = longest_dim(data)

    centered = data - data.mean(dim)
    # When the complex-valued points lie on a line that crosses the origin,
    # squaring them puts them all in the same quadrature. We can then take their
    # mean and then use *half* of the angle of that point (since squaring
    # doubles all angles) as the projection angle.
    angle = cast(
        xr.DataArray | xr.Dataset,
        xr.apply_ufunc(np.angle, (centered**2).mean(dim)) * 0.5,
    )
    result = cast(
        xr.DataArray | xr.Dataset,
        np.exp(-1j * angle) * centered,
    )

    if discard_imag:
        result = result.real

    return result


def simple_dft(
    data: xr.DataArray,
    dim: str,
    frequency_dim_name: str = "frequency",
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> xr.DataArray:
    """
    Simple discrete Fourier transform of a DataArray

    This is a simple wrapper around the `NumPy fft function
    <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html>`_. For
    more advanced use, see the `xrft <https://xrft.readthedocs.io/>`_ package.

    Assumes that the transform dimension is evenly spaced.

    Args:
        data: Data array to transform
        dim: Dimension along which to apply transform, usually time
        frequency_dim_name: Name of the transformed dimension in the output data.
        norm: Normalization mode, see the `numpy documentation <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html>`_ for details.
    """
    t = data[dim]
    if t.size < 2:
        raise ValueError(
            f"Dimension '{dim}' must have at least 2 points for DFT, got {t.size}"
        )
    dt = (t[1] - t[0]).item()
    f = np.fft.fftfreq(t.size, d=dt)

    spec = xr.apply_ufunc(
        np.fft.fft,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[frequency_dim_name]],
        kwargs={"norm": norm},
    )

    spec = spec.assign_coords({frequency_dim_name: f})
    spec = spec.sortby(frequency_dim_name)
    return spec
