"""
Functions for digital signal processing with xarray

These should complement functions available in the base xarray library and in
other external libraries such as xr-scipy or xrft. We should prefer those
whenever the functionality we need is available.
"""

from typing import cast, overload

import numpy as np
import xarray as xr
from xarray.core.types import Dims

from sqe_analysis.xr_util import longest_dim


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
    A simple way to project complex-valued data to the real axis in a way that maximizes the signal

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
