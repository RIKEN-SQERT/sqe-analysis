"""
Tests for TimeOfFlightanalysis
"""

import json
import warnings

import pytest

from sqe_fitting2.analysis import TimeOfFlightAnalysis
from sqe_fitting2.example_data import get_dataset_names
from sqe_fitting2.example_data import open_dataset as open_example_dataset
from sqe_fitting2.xr_util import longest_dim


# TODO: change to global fixture
def to_SI(value: float, units: str, dim: str, ds_name: str):
    if units == "us":
        return value * 1e6
    elif units == "ns":
        return value * 1e9
    else:
        raise ValueError(
            f"unknown units {units!r} on dimension {dim} in dataset {ds_name}"
        )


def test_time_of_flight_analysis_success():
    ds_names = [
        ds_name
        for ds_name in get_dataset_names()
        if ds_name.startswith("time_of_flight")
    ]

    for ds_name in ds_names:
        ds = open_example_dataset(ds_name)
        ds = ds.assign_attrs(expected_fit_result=json.loads(ds.expected_fit_result))
        dim = longest_dim(ds)
        units = ds[dim].units

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
