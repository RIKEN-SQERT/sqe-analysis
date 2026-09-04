import numpy as np
import xarray as xr
from xarray.testing import (  # pyright: ignore[reportUnknownVariableType]
    assert_allclose,
    assert_equal,
)

from sqe_analysis.example_data import get_dataset_names, open_dataset
from sqe_analysis.signal_processing import project_complex


def test_project_complex_dataarray_zero():
    da = xr.DataArray(np.array([0, 0]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([0, 0], dims=["x"]))


def test_project_complex_dataarray_real():
    da = xr.DataArray(np.array([0, 2]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([-1, 1], dims=["x"]))


def test_project_complex_dataarray_imag():
    da = xr.DataArray(np.array([-1j, 1j]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([-1, 1], dims=["x"]))


def test_project_complex_dataarray_mixed():
    da = xr.DataArray(np.array([-3 - 4j, 3 + 4j]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([-5, 5], dims=["x"]))


def test_project_complex_dataset_zero():
    ds = xr.Dataset({"v": (["x"], np.array([0, 0]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [0, 0])}))


def test_project_complex_dataset_real():
    ds = xr.Dataset({"v": (["x"], np.array([0, 2]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [-1, 1])}))


def test_project_complex_dataset_imag():
    ds = xr.Dataset({"v": (["x"], np.array([-1j, 1j]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [-1, 1])}))


def test_project_complex_dataset_mixed():
    ds = xr.Dataset({"v": (["x"], np.array([-3 - 4j, 3 + 4j]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [-5, 5])}))


def test_project_complex_2d_tuple_dim():
    """Project 2D complex data across both dimensions simultaneously."""
    da = xr.DataArray(
        # The data forms a triangle in the complex plane. Projecting across both
        # dimensions should yeld points at +-1. Projecting along a
        # single dimension would yield points at +- sqrt(5) / 2.
        np.array([[-1, 2j], [1, 2j]]),
        dims=["x", "y"],
    )
    projected = project_complex(da, dim=("x", "y"))
    assert_allclose(
        projected,
        xr.DataArray([[-1, 1], [-1, 1]], dims=["x", "y"]),
    )


def test_project_complex_idempotent():
    for ds_name in get_dataset_names():
        ds = open_dataset(ds_name)
        p = project_complex(ds)
        pp = project_complex(p)
        assert_allclose(p, pp)
