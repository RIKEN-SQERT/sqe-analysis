"""Ranked peak/dip detection retains sample indices and outer dimensions."""

import numpy as np
import pytest
import xarray as xr

from sqe_analysis.signal_processing import detect_peaks


def test_ranked_peaks_and_dips_with_padding():
    x = np.arange(9.0)
    y = xr.DataArray(
        [[0, 1, 0, 3, 0, 2, 0, 0, 0], [0, -1, 0, -3, 0, -2, 0, 0, 0]],
        dims=["trace", "x"],
        coords={"x": x},
        attrs={"dataset_id": "peaks"},
    )
    result = detect_peaks(y, "x", rank=4)
    np.testing.assert_array_equal(
        result["index"].values, [[3, 5, 1, -1], [3, 5, 1, -1]]
    )
    np.testing.assert_array_equal(result["is_dip"], [False, True])
    assert result["count"].values.tolist() == [3, 3]
    assert np.isnan(result.position.values[:, 3]).all()
    assert result.peak_mask.sum("x").values.tolist() == [3, 3]
    assert result.attrs["dataset_id"] == "peaks"


def test_flat_and_missing_data_do_not_create_false_peaks():
    y = xr.DataArray(
        [0, 1, np.nan, 3, 0, 0, np.inf, 0, 0], dims="x", coords={"x": np.arange(9)}
    )
    result = detect_peaks(y, "x", dip=False)
    assert int(result["count"]) == 0
    flat = detect_peaks(xr.zeros_like(y), "x")
    assert int(flat["count"]) == 0
    with pytest.raises(ValueError):
        detect_peaks(y, "x", rank=0)
    with pytest.raises(ValueError):
        detect_peaks(y + 1j, "x")
