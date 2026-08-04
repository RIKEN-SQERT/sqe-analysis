import numpy as np
import pytest
import xarray as xr

from sqe_fitting2.xr_util import longest_dim, sorted_dims


def test_sorted_dims_single_dimension():
    da = xr.DataArray(np.zeros(5), dims=["x"])
    assert sorted_dims(da) == ["x"]


def test_sorted_dims_two_dimensions_shorter_first():
    da = xr.DataArray(np.zeros((3, 7)), dims=["x", "y"])
    assert sorted_dims(da) == ["x", "y"]


def test_sorted_dims_two_dimensions_longer_first():
    da = xr.DataArray(np.zeros((9, 2)), dims=["x", "y"])
    assert sorted_dims(da) == ["y", "x"]


def test_sorted_dims_three_dimensions_mixed_order():
    da = xr.DataArray(np.zeros((5, 1, 10)), dims=["x", "y", "z"])
    assert sorted_dims(da) == ["y", "x", "z"]


def test_sorted_dims_equal_sizes_preserves_input_order():
    da = xr.DataArray(np.zeros((4, 4)), dims=["b", "a"])
    result = sorted_dims(da)
    # Both dims have size 4; sorted is stable so original order stays
    assert result == ["b", "a"]


def test_sorted_dims_dataset():
    ds = xr.Dataset({"v": (["x", "y"], np.zeros((6, 3)))})
    assert sorted_dims(ds) == ["y", "x"]


def test_sorted_dims_dataset_multiple_vars_same_dims():
    ds = xr.Dataset(
        {
            "a": (["x"], np.zeros(2)),
            "b": (["y"], np.zeros(8)),
        }
    )
    assert sorted_dims(ds) == ["x", "y"]


def test_longest_dim_single_dimension():
    da = xr.DataArray(np.zeros(5), dims=["x"])
    assert longest_dim(da) == "x"


def test_longest_dim_two_dimensions():
    da = xr.DataArray(np.zeros((3, 7)), dims=["x", "y"])
    assert longest_dim(da) == "y"


def test_longest_dim_three_dimensions():
    da = xr.DataArray(np.zeros((5, 1, 10)), dims=["x", "y", "z"])
    assert longest_dim(da) == "z"


def test_longest_dim_equal_sizes_returns_last_after_stable_sort():
    da = xr.DataArray(np.zeros((4, 4)), dims=["b", "a"])
    assert longest_dim(da) == "a"


def test_longest_dim_dataset():
    ds = xr.Dataset({"v": (["x", "y"], np.zeros((6, 3)))})
    assert longest_dim(ds) == "x"


def test_longest_dim_zero_dimensional_raises():
    da = xr.DataArray(42)
    with pytest.raises(ValueError, match="zero-dimensional"):
        longest_dim(da)  # pyright: ignore[reportUnusedCallResult]
