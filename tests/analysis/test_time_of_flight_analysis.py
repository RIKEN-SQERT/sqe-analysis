"""
Tests for TimeOfFlightanalysis
"""

import warnings

import pytest
import xarray as xr
from util import open_test_dataset, to_SI

from sqe_analysis.analysis import TimeOfFlightAnalysis
from sqe_analysis.example_data import get_dataset_names


def test_time_of_flight_analysis_basic():
    data = xr.DataArray(
        [
            # small offsets so we don't get infinite SNR
            -0.01,
            0.01,
            -0.01,
            0.01,
            1 - 0.01,
            1 + 0.01,
            1 - 0.01,
            1 + 0.01,
        ],
        coords=[("x", list(range(8)))],
        attrs={"dataset_id": "test"},
    )
    result = TimeOfFlightAnalysis.run(data)
    assert result.params.SNR.item() > 100
    assert result.params.step_location.item() == 3
    assert result.success.item()


def test_time_of_flight_analysis_success():
    ds_names = [
        ds_name
        for ds_name in get_dataset_names()
        if ds_name.startswith("time_of_flight")
    ]

    for ds_name in ds_names:
        ds, dim, units = open_test_dataset(ds_name)

        result = TimeOfFlightAnalysis.run(ds.to_dataarray(dim="qubit"))
        for q in ds.data_vars:
            if (ds_name, q) in [
                ("time_of_flight-medium_snr_full_pulse-RX4_20260803", "Q32"),
                ("time_of_flight-medium_snr_full_pulse-RX4_20260803", "Q48"),
            ]:
                warnings.warn(
                    f"Skipping {ds_name} {q}, analysis should be improved so that this is not necessary"
                )
                continue

            expected = ds.expected_fit_result[q]["rising_edge"]
            expected = to_SI(expected, units=units, dim=dim, ds_name=ds_name)

            actual = result.params.step_location.sel(qubit=q).item()

            assert result.success.sel(qubit=q).item(), (ds_name, q)
            assert actual == pytest.approx(expected, rel=0.05)
