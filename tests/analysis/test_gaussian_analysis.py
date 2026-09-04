"""
Tests for GaussianAnalysis
"""

import numpy as np
import pytest
import xarray as xr
from util import open_test_dataset
from xarray.testing import assert_allclose
from xarray.testing.assertions import assert_equal

from sqe_analysis.analysis import GaussianAnalysis
from sqe_analysis.signal_processing import project_complex


def test_gaussian_analysis_stark_shift_high_snr():
    data, _, _ = open_test_dataset(
        "ac_stark_shift_vs_resonator_frequency-high_snr-RX3_1757"
    )
    data = data.Q72.assign_attrs(dataset_id=data.dataset_id)

    # shorthands for dimension names
    f_q = "qubit_drive_frequency_shift"
    f_r = "resonator_drive_frequency_shift"

    # use minimum location as approximate expected fit result
    expected = (
        data.pipe(project_complex, dim=f_q)
        .pipe(lambda y: y - y.median([f_q, f_r]))
        .pipe(abs)
        .idxmax(f_q)
    )

    result = GaussianAnalysis.run(data, coords=f_q)

    assert result.success.all()

    c = result.params.c

    # The step size is around 3MHz, the range of values is around -80 to +160
    # MHz and the linewidth is between 10..35 MHz, so a tolerance of a few MHz
    # is acceptable.

    flat_region = slice(None, -20)
    assert_allclose(
        c.sel({f_r: flat_region}),
        expected.sel({f_r: flat_region}),
        atol=5.0,
    )

    # close to the resonance the linewidth is broader so increase the tolerance
    near_resonance = slice(-20, 10)
    assert_allclose(
        c.sel({f_r: near_resonance}),
        expected.sel({f_r: near_resonance}),
        atol=10.0,
    )

    # check that fit worked for both dip & peak
    assert_equal(
        np.sign(result.params.a.isel({f_r: 0}, drop=True)),
        xr.DataArray([-1, 1], coords=[result.params.qubit_pulse_amplitude]),
    )


def test_gaussian_analysis_partial_half_curve():
    # fit when there is a large baseline but only half of the bell curve is visible
    data, _, _ = open_test_dataset(
        "ac_stark_shift_vs_resonator_frequency-high_snr-RX3_1757"
    )
    f_q = "qubit_drive_frequency_shift"
    data = (
        data.Q72.assign_attrs(dataset_id=data.dataset_id)
        .isel(
            resonator_drive_frequency_shift=0,
            qubit_pulse_amplitude=0,
        )
        .sel({f_q: slice(None, 4)})
    )

    result = GaussianAnalysis.run(data, coords=f_q)
    assert result.params.c.item() == pytest.approx(0, abs=1.0)


def test_gaussian_analysis_partial_less_than_half_curve():
    # fit when there is a large baseline but less than half of the bell curve is visible
    data, _, _ = open_test_dataset(
        "ac_stark_shift_vs_resonator_frequency-high_snr-RX3_1757"
    )
    f_q = "qubit_drive_frequency_shift"
    data = (
        data.Q72.assign_attrs(dataset_id=data.dataset_id)
        .isel(
            resonator_drive_frequency_shift=0,
            qubit_pulse_amplitude=0,
        )
        .sel({f_q: slice(None, -10)})
    )

    result = GaussianAnalysis.run(data, coords=f_q)
    # note: the estimate is off with this choice of limits but the fit is still good
    assert result.params.c.item() == pytest.approx(-5.0, abs=0.2)


@pytest.mark.xfail(
    reason="GaussianAnalysis is not robust against partially visible bell curve, should be improved"
)
def test_gaussian_analysis_partial_middle_curve():
    # fit when only the middle part (~between FWHM) is visible
    data, _, _ = open_test_dataset(
        "ac_stark_shift_vs_resonator_frequency-high_snr-RX3_1757"
    )
    f_q = "qubit_drive_frequency_shift"
    data = (
        data.Q72.assign_attrs(dataset_id=data.dataset_id)
        .isel(
            resonator_drive_frequency_shift=0,
            qubit_pulse_amplitude=0,
        )
        .sel({f_q: slice(-15, 15)})
    )

    result = GaussianAnalysis.run(data, coords=f_q)
    assert result.success.all()
    assert result.params.c.item() == pytest.approx(0, abs=0.5)
    assert "FWHM" in result.params


@pytest.mark.xfail(
    reason="TODO: add proper failure criterion to curve fit based on chi squared / SNR threshold"
)
def test_gaussian_analysis_no_signal():
    data, _, _ = open_test_dataset(
        "ac_stark_shift_vs_resonator_frequency-high_snr-RX3_1757"
    )
    f_q = "qubit_drive_frequency_shift"
    data = (
        data.Q72.assign_attrs(dataset_id=data.dataset_id)
        .isel(
            resonator_drive_frequency_shift=0,
            qubit_pulse_amplitude=0,
        )
        .sel({f_q: slice(None, -50)})
    )

    result = GaussianAnalysis.run(data, f_q)

    assert not result.success.all()
