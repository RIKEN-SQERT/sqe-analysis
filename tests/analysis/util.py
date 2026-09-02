"""
Utility functions for the data analysis tests
"""

import json

import xarray as xr

from sqe_analysis.example_data import open_dataset as open_example_dataset
from sqe_analysis.xarray_util import longest_dim


def to_SI(value: float, units: str, dim: str, ds_name: str) -> float:
    """Quick helper function for converting dimension units to SI"""
    if units == "us":
        return value * 1e6
    elif units == "ns":
        return value * 1e9
    else:
        raise ValueError(
            f"unknown units {units!r} on dimension {dim} in dataset {ds_name}"
        )


def open_test_dataset(
    ds_name: str, dim: str | None = None
) -> tuple[xr.Dataset, str, str]:
    ds = open_example_dataset(ds_name)
    ds = ds.assign_attrs(
        dataset_id=ds.source,
    )

    if "expected_fit_result" in ds.attrs:
        ds = ds.assign_attrs(
            expected_fit_result=json.loads(ds.expected_fit_result),
        )

    if dim is None:
        dim = longest_dim(ds)
    units = ds[dim].units

    return ds, dim, units
