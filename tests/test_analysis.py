"""
Testing of the data analysis methods, using the real example data
"""

import xarray as xr

from sqe_fitting2.analysis_base import CurvefitAnalysis


def test_curvefit_analysis_basic():
    """
    Trivial curvefit analysis test
    """

    class LineFit(CurvefitAnalysis):
        def func(x, a, b):
            return a * x + b

    res = LineFit.run(
        xr.DataArray([-1, 1, 3], coords=[("x", [0, 1, 2])]),
        coords="x",
    )

    assert res.params.a == 2
    assert res.params.b == -1


def test_curvefit_analysis_with_guess_method():
    pass


def test_curvefit_analysis_with_guess_arg():
    pass
