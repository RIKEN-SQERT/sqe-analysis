"""
Small utility functions for dealing with Xarray arrays
"""

from typing import cast

import xarray as xr


def sorted_dims(data: xr.DataArray | xr.Dataset) -> list[str]:
    """
    Return the names of the dimensions of a Dataset or DataArray, with the
    shortest dimension first.
    """
    kvs = sorted(data.sizes.items(), key=lambda kv: kv[1])
    # Cast dimensions to str to make type checking simpler. In principle, the
    # dimensions can also be tuples, but this is very rare. We can figure out
    # the correct type hints if we actually encounter a situation where we need
    # tuples.
    return [cast(str, kv[0]) for kv in kvs]


def longest_dim(data: xr.DataArray | xr.Dataset) -> str:
    """
    Return the longest dimension of the Dataset or DataArray. If the data is
    zero-dimensional, raises a ValueError.
    """
    dims = sorted_dims(data)
    if len(dims) == 0:
        raise ValueError("Trying to get longest dimension of zero-dimensional data")
    return dims[-1]
