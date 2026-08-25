"""
Testing of the analysis result classes
"""

import xarray as xr

from sqe_fitting2.result import AnalysisResult


def test_result_html_repr():
    r = AnalysisResult(
        params=xr.Dataset(
            {
                "foo": 123,
                "bar": 456,
            }
        ),
        success=xr.DataArray(True),
    )

    # should not raise an exception
    r._repr_html_()
