"""
Small utility functions for dealing with xarray arrays
"""

from collections.abc import Hashable

import xarray as xr


def sorted_dims(data: xr.DataArray | xr.Dataset) -> list[Hashable]:
    """
    Return the names of the dimensions of a Dataset or DataArray, with the
    shortest dimension first.
    """
    kvs = sorted(data.sizes.items(), key=lambda kv: kv[1])
    return [kv[0] for kv in kvs]


def longest_dim(data: xr.DataArray | xr.Dataset) -> Hashable:
    """
    Return the longest dimension of the Dataset or DataArray. If the data is
    zero-dimensional, raises a ValueError.
    """
    dims = sorted_dims(data)
    if len(dims) == 0:
        raise ValueError("Trying to get longest dimension of zero-dimensional data")
    return dims[-1]
