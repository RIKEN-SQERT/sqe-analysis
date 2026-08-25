"""
Testing of the data analysis methods, using the real example data
"""

import pytest
import xarray as xr
from util import open_test_dataset, to_SI

from sqe_fitting2.analysis import ExponentialRegressionAnalysis
from sqe_fitting2.example_data import get_dataset_names


def test_exponential_regression_analysis_basic():
    data = (
        xr.DataArray(
            [1, 0.368, 0.135],
            coords=[("x", [0, 1, 2])],
            attrs={"dataset_id": "test"},
        )
        * 2
        + 3
    )
    result = ExponentialRegressionAnalysis.run(data)
    assert result.params.k.item() == pytest.approx(1, rel=0.1)
    assert result.params.a.item() == pytest.approx(2, rel=0.1)
    assert result.params.b.item() == pytest.approx(3, rel=0.1)
    assert result.source_dataset_id == "test"
    assert ExponentialRegressionAnalysis.__name__ in result.analysis_class


def test_exponential_regression_analysis_t1():
    ds_names = [ds_name for ds_name in get_dataset_names() if ds_name.startswith("t1-")]
    for ds_name in ds_names:
        ds, dim, units = open_test_dataset(ds_name)

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
                ("t1-high_snr_and_no_signal-RX4_QM_30", "Q40"),
                ("t1-high_snr_and_no_signal-RX4_QM_30", "Q42"),
                ("t1-high_snr_and_no_signal-RX4_QM_30", "Q43"),
                ("t1-high_low_snr_and_no_signal_cut_off-RX4_QM_29", "Q40"),
                ("t1-high_low_snr_and_no_signal_cut_off-RX4_QM_29", "Q42"),
                # TODO: decent SNR but initial data points are invalid / outliers
                ("t1-high_low_snr_and_no_signal_cut_off-RX4_QM_29", "Q43"),
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
            expected = to_SI(expected, units=units, dim=dim, ds_name=ds_name)

            actual = fit_result.params.decay_constant.sel(qubit=q).item()
            assert actual == pytest.approx(expected, rel=0.21, abs=1.8), (ds_name, q)

    # TODO: tests for 'no signal' case


def test_exponential_regression_analysis_t1_flip():
    """
    Test for ExponentialRegressionAnalysis on t1_flip-* data. No simultaneous
    fit, so might get different time constants for the different values of
    pi_pulse_at_end
    """
    ds_names = [
        ds_name for ds_name in get_dataset_names() if ds_name.startswith("t1_flip-")
    ]
    for ds_name in ds_names:
        ds, dim, units = open_test_dataset(ds_name)

        fit_result = ExponentialRegressionAnalysis.run(
            ds.to_dataarray(dim="qubit"), dim=dim
        )

        for q in ds.data_vars:
            if (ds_name, q) in [
                ("t1_flip-distorted-RX4_59", "Q31"),  # bad fit in original dataset
                ("t1_flip-no_signal-RX4_59", "Q44"),
                ("t1_flip-no_signal-RX4_59", "Q47"),
            ]:
                continue
            expected = ds.expected_fit_result[q]["t1"]
            expected = to_SI(expected, units=units, dim=dim, ds_name=ds_name)

            # just take the mean of the two values for now
            actual = (
                fit_result.params.decay_constant.sel(qubit=q)
                .mean("pi_pulse_at_end")
                .item()
            )
            assert actual == pytest.approx(expected, rel=0.25), (ds_name, q)
            assert fit_result.success.sel(qubit=q).all().item(), (ds_name, q)


def test_exponential_regression_analysis_t1_flip_failures():
    for ds_name, qs in {
        "t1-high_snr_and_no_signal-RX4_QM_30": [
            "Q40",
            "Q42",
            "Q43",
        ],
        "t1-high_low_snr_and_no_signal_cut_off-RX4_QM_29": ["Q40", "Q42"],
        "t1_flip-no_signal-RX4_59": ["Q44", "Q47"],
    }.items():
        ds, dim, _ = open_test_dataset(ds_name)

        fit_result = ExponentialRegressionAnalysis.run(
            ds.to_dataarray(dim="qubit"), dim=dim
        )

        for q in qs:
            assert not fit_result.success.sel(qubit=q).all()
