"""
Testing of the data analysis base classes
"""

from typing import override

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from sqe_analysis.analysis_base import CurvefitAnalysis
from sqe_analysis.result import CurvefitAnalysisResult

# ---------------------------------------------------------------------------
# CurvefitAnalysis
# ---------------------------------------------------------------------------


class LineFit(CurvefitAnalysis):
    @classmethod
    def func(cls, x, a, b):
        return a * x + b


class LineFitWithGuess(CurvefitAnalysis):
    @classmethod
    def func(cls, x, a, b):
        return a * x + b

    @classmethod
    def guess(cls, preprocessed_data):
        return {"a": 0.0, "b": 0.0}


class LineFitWithPreprocess(CurvefitAnalysis):
    """Subtracts a baseline offset before fitting."""

    @classmethod
    def func(cls, x, a, b):
        return a * x + b

    @classmethod
    def preprocess(cls, data, coords):
        return data - 100


class LineFitWithDerivedQuantity(CurvefitAnalysis):
    @classmethod
    def func(cls, x, a, b):
        return a * x + b

    @classmethod
    @override
    def extra_params(cls, fit_params: xr.Dataset) -> xr.Dataset | None:
        return xr.Dataset({"x_intercept": -fit_params.b / fit_params.a})


class QuadraticFit(CurvefitAnalysis):
    @classmethod
    def func(cls, x, a, b, c):
        return a * x**2 + b * x + c


# ---------------------------------------------------------------------------
# Basic fitting
# ---------------------------------------------------------------------------


def test_curvefit_analysis_basic():
    """Trivial curvefit analysis test."""

    res = LineFit.run(
        xr.DataArray(
            [-1, 1, 3], coords=[("x", [0, 1, 2])], attrs={"dataset_id": "test"}
        ),
        coords="x",
    )

    assert isinstance(res, CurvefitAnalysisResult)
    assert res.params.a.item() == 2.0
    assert res.params.b.item() == -1.0
    assert res.fit_params.a.item() == 2.0
    assert res.fit_params.b.item() == -1.0
    assert res.success.coords == res.params.coords


# ---------------------------------------------------------------------------
# CurvefitAnalysis guess
# ---------------------------------------------------------------------------


def test_curvefit_analysis_with_guess_method():
    """The guess method should provide initial values used by the fit."""

    res = LineFitWithGuess.run(
        xr.DataArray(
            [1, 3, 5], coords=[("x", [0, 1, 2])], attrs={"dataset_id": "test"}
        ),
        coords="x",
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)
    assert res.fit_params.a.item() == pytest.approx(2.0)
    assert res.fit_params.b.item() == pytest.approx(1.0)
    assert res.fit_params_guess.a.item() == 0
    assert res.fit_params_guess.b.item() == 0


def test_curvefit_analysis_with_guess_arg():
    """A guess argument should override the guess method values."""

    # Method returns a=0, b=0; arg overrides a=1.0 only.
    res = LineFitWithGuess.run(
        xr.DataArray(
            [1, 3, 5], coords=[("x", [0, 1, 2])], attrs={"dataset_id": "test"}
        ),
        coords="x",
        guess={"a": 1.0},
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)
    assert res.fit_params.a.item() == pytest.approx(2.0)
    assert res.fit_params.b.item() == pytest.approx(1.0)
    assert res.fit_params_guess.a.item() == 1
    assert res.fit_params_guess.b.item() == 0


def test_curvefit_analysis_guess_arg_without_method():
    """A guess argument works even when the class has no custom guess method."""

    res = LineFit.run(
        xr.DataArray(
            [1, 3, 5], coords=[("x", [0, 1, 2])], attrs={"dataset_id": "test"}
        ),
        coords="x",
        guess={"a": 1.0, "b": 0.0},
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)
    assert res.fit_params.a.item() == pytest.approx(2.0)
    assert res.fit_params.b.item() == pytest.approx(1.0)
    assert res.fit_params_guess.a.item() == 1.0
    assert res.fit_params_guess.b.item() == 0.0


def test_curvefit_analysis_guess_arg_without_method_partial():
    """
    A guess argument works even when the class has no custom guess method.
    Only the parameters given in the arguments should show up in the output.
    """

    res = LineFit.run(
        xr.DataArray(
            [1, 3, 5], coords=[("x", [0, 1, 2])], attrs={"dataset_id": "test"}
        ),
        coords="x",
        guess={"a": 1.0},
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)
    assert res.fit_params.a.item() == pytest.approx(2.0)
    assert res.fit_params.b.item() == pytest.approx(1.0)
    assert res.fit_params_guess.a.item() == 1.0
    assert list(res.fit_params_guess.keys()) == ["a"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_curvefit_analysis_preprocessing():
    """Preprocessing should modify the data before fitting."""
    # Raw data is y = 2x + 1 + 100. Preprocess subtracts 100.
    data = xr.DataArray(
        [101, 103, 105],
        coords=[("x", [0, 1, 2])],
        dims=["x"],
        attrs={"dataset_id": "test"},
    )

    res = LineFitWithPreprocess.run(data, coords="x")

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)
    assert res.fit_params.a.item() == pytest.approx(2.0)
    assert res.fit_params.b.item() == pytest.approx(1.0)
    assert_allclose(
        res.intermediate_results.preprocessed_data,
        xr.DataArray([1, 3, 5], coords=[data.x]),
    )


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def test_curvefit_analysis_derived_quantities():
    res = LineFitWithDerivedQuantity.run(
        xr.DataArray(
            [1, 3, 5], coords=[("x", [0, 1, 2])], attrs={"dataset_id": "test"}
        ),
        coords="x",
        guess={"a": 1.0},
    )

    assert list(res.params.keys()) == ["a", "b", "x_intercept"]
    assert list(res.fit_params.keys()) == ["a", "b"]

    assert res.params.a.item() == 2.0
    assert res.params.b.item() == 1.0
    assert res.params.x_intercept == -0.5
    assert res.fit_params.a.item() == 2.0
    assert res.fit_params.b.item() == 1.0


# ---------------------------------------------------------------------------
# Multidimensional fitting
# ---------------------------------------------------------------------------


def test_curvefit_analysis_multidimensional():
    """Fitting along one dimension with multiple independent traces."""
    data = xr.DataArray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [-1.0, 1.0, 3.0, 5.0, 7.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        coords=[("trace", ["a", "b", "c"]), ("x", [0, 1, 2, 3, 4])],
        attrs={"dataset_id": "test"},
    )

    res = LineFit.run(data, coords="x")

    assert_allclose(res.params.a, xr.DataArray([1, 2, -1], coords=[data.trace]))
    assert_allclose(res.params.b, xr.DataArray([0, -1, 5], coords=[data.trace]))
    assert_allclose(res.fit_params.a, xr.DataArray([1, 2, -1], coords=[data.trace]))
    assert_allclose(res.fit_params.b, xr.DataArray([0, -1, 5], coords=[data.trace]))


def test_curvefit_analysis_multidimensional_guess():
    """Guess can be a scalar (same for all traces) or per-trace."""

    data = xr.DataArray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [-1.0, 1.0, 3.0, 5.0, 7.0],
        ],
        coords=[("trace", ["a", "b"]), ("x", [0, 1, 2, 3, 4])],
        attrs={"dataset_id": "test"},
    )

    # Scalar guess broadcast to all traces
    res = LineFit.run(
        data,
        coords="x",
        guess={"a": 0.0, "b": xr.DataArray([0, -1], coords=[data.trace])},
    )

    assert_allclose(res.params.a, xr.DataArray([1, 2], coords=[data.trace]))
    assert_allclose(res.fit_params.a, xr.DataArray([1, 2], coords=[data.trace]))
    assert res.fit_params_guess.a.item() == 0
    assert_allclose(res.fit_params_guess.b, xr.DataArray([0, -1], coords=[data.trace]))


def test_curvefit_analysis_quadratic():
    """Fitting a more complex model function."""
    # y = 2x^2 + 3x + 1
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x**2 + 3 * x + 1

    res = QuadraticFit.run(
        xr.DataArray(y, coords=[("x", x)], dims=["x"], attrs={"dataset_id": "test"}),
        coords="x",
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(3.0)
    assert res.params.c.item() == pytest.approx(1.0)
    assert res.fit_params.a.item() == pytest.approx(2.0)
    assert res.fit_params.b.item() == pytest.approx(3.0)
    assert res.fit_params.c.item() == pytest.approx(1.0)
