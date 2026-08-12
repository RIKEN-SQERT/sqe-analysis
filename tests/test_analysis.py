"""
Testing of the data analysis methods, using the real example data
"""

import numpy as np
import pytest
import xarray as xr

from sqe_fitting2.analysis_base import BaseAnalysis, CurvefitAnalysis
from sqe_fitting2.result import AnalysisResult, CurvefitAnalysisResult

# ---------------------------------------------------------------------------
# BaseAnalysis
# ---------------------------------------------------------------------------


def test_base_analysis_run_raises():
    """BaseAnalysis.run should raise NotImplementedError."""
    data = xr.DataArray([1, 2, 3])
    with pytest.raises(NotImplementedError, match="Analysis not implemented"):
        BaseAnalysis.run(data)


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
    def guess(cls, data):
        return {"a": 0.0, "b": 0.0}


class LineFitWithPreprocess(CurvefitAnalysis):
    """Subtracts a baseline offset before fitting."""

    @classmethod
    def func(cls, x, a, b):
        return a * x + b

    @classmethod
    def preprocess(cls, data, coords):
        return data - 100


class QuadraticFit(CurvefitAnalysis):
    @classmethod
    def func(cls, x, a, b, c):
        return a * x**2 + b * x + c


def test_curvefit_func_raises_for_abstract():
    """Calling CurvefitAnalysis.func directly should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Model function not implemented"):
        CurvefitAnalysis.func(0)


def test_curvefit_guess_default_returns_none():
    """Default guess method should return None."""
    assert CurvefitAnalysis.guess(xr.DataArray([1])) is None


def test_curvefit_preprocess_default_returns_none():
    """Default preprocess method should return None."""
    assert CurvefitAnalysis.preprocess(xr.DataArray([1]), coords="x") is None


# ---------------------------------------------------------------------------
# Basic fitting
# ---------------------------------------------------------------------------


def test_curvefit_analysis_basic():
    """Trivial curvefit analysis test."""

    res = LineFit.run(
        xr.DataArray([-1, 1, 3], coords=[("x", [0, 1, 2])]),
        coords="x",
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Guess: method and argument
# ---------------------------------------------------------------------------


def test_curvefit_analysis_with_guess_method():
    """The guess method should provide initial values used by the fit."""

    res = LineFitWithGuess.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)


def test_curvefit_analysis_with_guess_arg():
    """A guess argument should override the guess method values."""

    # Method returns a=0, b=0; arg overrides a=1.0 only.
    res = LineFitWithGuess.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
        guess={"a": 1.0},
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)


def test_curvefit_analysis_guess_arg_overrides_method():
    """When both method and arg provide a guess, arg values take priority."""

    # Method returns a=0, b=0. Arg overrides both.
    res = LineFitWithGuess.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
        guess={"a": 1.5, "b": 0.5},
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)


def test_curvefit_analysis_guess_arg_without_method():
    """A guess argument works even when the class has no custom guess method."""

    res = LineFit.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
        guess={"a": 1.0, "b": 0.0},
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_curvefit_analysis_preprocessing():
    """Preprocessing should modify the data before fitting."""
    # Raw data is y = 2x + 1 + 100. Preprocess subtracts 100.
    data = xr.DataArray([101, 103, 105], coords=[("x", [0, 1, 2])], dims=["x"])

    res = LineFitWithPreprocess.run(data, coords="x")

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(1.0)


def test_curvefit_analysis_preprocessing_stores_intermediate():
    """Preprocessed data should be available in intermediate_results."""
    data = xr.DataArray([101, 103, 105], coords=[("x", [0, 1, 2])], dims=["x"])

    res = LineFitWithPreprocess.run(data, coords="x")

    assert res.intermediate_results is not None
    assert "preprocessed_data" in res.intermediate_results


def test_curvefit_analysis_no_preprocessing_no_intermediate():
    """Without preprocessing, intermediate_results should be None."""
    res = LineFit.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
    )

    assert res.intermediate_results is None


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
    )

    res = LineFit.run(data, coords="x")

    # trace a: a=1, b=0
    assert res.params.a.sel(trace="a").item() == pytest.approx(1.0)
    assert res.params.b.sel(trace="a").item() == pytest.approx(0.0)

    # trace b: a=2, b=-1
    assert res.params.a.sel(trace="b").item() == pytest.approx(2.0)
    assert res.params.b.sel(trace="b").item() == pytest.approx(-1.0)

    # trace c: a=-1, b=5
    assert res.params.a.sel(trace="c").item() == pytest.approx(-1.0)
    assert res.params.b.sel(trace="c").item() == pytest.approx(5.0)


def test_curvefit_analysis_multidimensional_guess():
    """Guess can be a scalar (same for all traces) or per-trace."""

    # Scalar guess broadcast to all traces
    res = LineFit.run(
        xr.DataArray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [-1.0, 1.0, 3.0, 5.0, 7.0],
            ],
            coords=[("trace", ["a", "b"]), ("x", [0, 1, 2, 3, 4])],
        ),
        coords="x",
        guess={"a": 0.0, "b": 0.0},
    )

    assert res.params.a.sel(trace="a").item() == pytest.approx(1.0)
    assert res.params.a.sel(trace="b").item() == pytest.approx(2.0)


def test_curvefit_analysis_quadratic():
    """Fitting a more complex model function."""
    # y = 2x^2 + 3x + 1
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x**2 + 3 * x + 1

    res = QuadraticFit.run(
        xr.DataArray(y, coords=[("x", x)], dims=["x"]),
        coords="x",
    )

    assert res.params.a.item() == pytest.approx(2.0)
    assert res.params.b.item() == pytest.approx(3.0)
    assert res.params.c.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


def test_analysis_result_params():
    """AnalysisResult.params should expose the params DatasetView."""
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    res = AnalysisResult(params=ds)
    assert res.params.a.sum().item() == 6.0


def test_analysis_result_optional_none():
    """Optional fields not provided should return None."""
    res = AnalysisResult(params=xr.Dataset({"a": 1}))
    assert res.params_std is None
    assert res.intermediate_results is None
    assert res.debug_results is None


def test_analysis_result_optional_provided():
    """Optional fields that are provided should be accessible."""
    res = AnalysisResult(
        params=xr.Dataset({"a": 1}),
        params_std=xr.Dataset({"a_std": 0.1}),
        intermediate_results=xr.Dataset({"histogram": ("x", [1, 2, 3])}),
        debug_results=xr.Dataset({"info": "test"}),
    )
    assert res.params_std is not None
    assert res.intermediate_results is not None
    assert res.debug_results is not None
    assert res.params_std.a_std.item() == pytest.approx(0.1)


def test_analysis_result_partial_optional():
    """Only some optional fields provided."""
    res = AnalysisResult(
        params=xr.Dataset({"a": 1}),
        params_std=xr.Dataset({"a_std": 0.1}),
    )
    assert res.params_std is not None
    assert res.intermediate_results is None
    assert res.debug_results is None


# ---------------------------------------------------------------------------
# CurvefitAnalysisResult
# ---------------------------------------------------------------------------


def test_curvefit_analysis_result_params_derived_none():
    """params_derived should be None when not provided."""
    res = CurvefitAnalysisResult(params=xr.Dataset({"a": 1}))
    assert res.params_derived is None


def test_curvefit_analysis_result_params_guess_none():
    """params_guess should be None when not provided."""
    res = CurvefitAnalysisResult(params=xr.Dataset({"a": 1}))
    assert res.params_guess is None


def test_curvefit_analysis_result_all_fields():
    """All CurvefitAnalysisResult fields should be accessible."""
    res = CurvefitAnalysisResult(
        params=xr.Dataset({"a": 1, "b": 2}),
        params_std=xr.Dataset({"a": 0.1}),
        params_derived=xr.Dataset({"r_squared": 0.99}),
        params_guess=xr.Dataset({"a": 0.5}),
        intermediate_results=xr.Dataset({"residual": ("x", [0.1])}),
        debug_results=xr.Dataset({"iterations": 10}),
    )

    assert res.params_std is not None
    assert res.params_derived is not None
    assert res.params_guess is not None
    assert res.intermediate_results is not None
    assert res.debug_results is not None

    assert res.params_derived.r_squared.item() == pytest.approx(0.99)
    assert res.params_guess.a.item() == pytest.approx(0.5)


def test_curvefit_analysis_result_inherits_analysis_result():
    """CurvefitAnalysisResult should inherit AnalysisResult properties."""
    res = CurvefitAnalysisResult(
        params=xr.Dataset({"a": 1}),
        params_std=xr.Dataset({"a": 0.1}),
        debug_results=xr.Dataset({"info": "test"}),
    )

    assert res.params is not None
    assert res.params_std is not None
    assert res.debug_results is not None
    assert res.params_derived is None
    assert res.params_guess is None


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_curvefit_run_returns_curvefit_analysis_result():
    """run() should return a CurvefitAnalysisResult, not just AnalysisResult."""
    res = LineFit.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
    )
    assert isinstance(res, CurvefitAnalysisResult)


def test_curvefit_result_has_params_derived_attribute():
    """The result should have params_derived and params_guess attributes."""
    res = LineFit.run(
        xr.DataArray([1, 3, 5], coords=[("x", [0, 1, 2])]),
        coords="x",
    )
    assert hasattr(res, "params_derived")
    assert hasattr(res, "params_guess")
    assert res.params_derived is None
    assert res.params_guess is None
