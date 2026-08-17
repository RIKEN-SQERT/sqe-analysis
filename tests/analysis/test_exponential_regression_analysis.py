"""
Testing of the data analysis methods, using the real example data
"""

import json

import pytest

from sqe_fitting2.analysis import ExponentialRegressionAnalysis
from sqe_fitting2.example_data import get_dataset_names
from sqe_fitting2.example_data import open_dataset as open_example_dataset
from sqe_fitting2.xr_util import longest_dim


def test_exponential_regression_analysis_t1():
    ds_names = [ds_name for ds_name in get_dataset_names() if ds_name.startswith("t1-")]
    for ds_name in ds_names:
        ds = open_example_dataset(ds_name)
        ds = ds.assign_attrs(expected_fit_result=json.loads(ds.expected_fit_result))
        dim = longest_dim(ds)
        units = ds[dim].units

        fit_result = ExponentialRegressionAnalysis.run(
            ds.to_dataarray(dim="qubit"), dim=dim
        )
        for q in ds.data_vars:
            # skip some bad data
            if (ds_name, q) in [
                ("t1-good_snr_very_short_t1-RX4_QD20260730016", "Q26"),
                ("t1-good_snr_very_short_t1-RX4_QD20260730016", "Q36"),
                ("t1-good_snr_very_short_t1-RX4_QD20260730016", "Q52"),
                ("t1-good_snr_very_short_t1-RX4_QD20260730016", "Q54"),
                ("t1-high_low_snr-RX4_QM_30", "Q40"),
                ("t1-high_low_snr-RX4_QM_30", "Q42"),
                ("t1-high_low_snr-RX4_QM_30", "Q43"),
                ("t1-high_low_snr_cut_off-RX4_QM_29", "Q40"),
                ("t1-high_low_snr_cut_off-RX4_QM_29", "Q42"),
                # TODO: decent SNR but initial data points are invalid / outliers
                ("t1-high_low_snr_cut_off-RX4_QM_29", "Q43"),
                ("t1-high_medium_snr_cut_off-RX4_QM_28", "Q40"),
                ("t1-high_medium_snr_cut_off-RX4_QM_28", "Q41"),
                ("t1-high_medium_snr_cut_off-RX4_QM_28", "Q42"),
                ("t1-high_medium_snr_cut_off-RX4_QM_28", "Q43"),
                ("t1-high_snr_cut_off-RX4_QM_33", "Q40"),
                ("t1-high_snr_cut_off-RX4_QM_33", "Q41"),
                ("t1-high_snr_cut_off-RX4_QM_33", "Q42"),
                ("t1-high_snr_cut_off-RX4_QM_33", "Q43"),
            ]:
                continue

            expected = ds.expected_fit_result[q]["t1"]
            if units == "us":
                expected = expected * 1e6
            elif units == "ns":
                expected = expected * 1e9
            else:
                raise ValueError(
                    f"unknown units {units} on dimension {dim} in dataset {ds_name}"
                )
            actual = fit_result.params.decay_constant.sel(qubit=q).item()
            assert actual == pytest.approx(expected, rel=0.21, abs=1.8), (ds_name, q)

    # TODO: tests for 'no signal' case


def test_exponential_regression_analysis_t1_flip():
    _ds_names = [
        ds_name for ds_name in get_dataset_names() if ds_name.startswith("t1_flip-")
    ]
    raise pytest.skip()
