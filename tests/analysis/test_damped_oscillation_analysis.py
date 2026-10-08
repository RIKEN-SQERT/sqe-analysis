"""
Tests for DampedOscillationAnalysis
"""

import numpy as np
import pytest
import xarray as xr
from util import open_test_dataset

from sqe_analysis.analysis import DampedOscillationAnalysis


@pytest.mark.parametrize("phase", [-0.2, 0.125, 0.8])
@pytest.mark.parametrize("first_time", [0.0, 3.0])
def test_damped_oscillation_phase_guess(phase, first_time):
    """Estimate phase in turns relative to time zero."""
    sample_count = 256
    time_step = 0.25

    # Use exactly seven periods over the DFT window.
    frequency = 7 / (sample_count * time_step)
    time = first_time + np.arange(sample_count) * time_step

    data = xr.DataArray(
        0.2 + 0.8 * np.cos(2 * np.pi * (frequency * time + phase)),
        coords={"time": time},
    )

    guess = DampedOscillationAnalysis.guess(data, coords="time")

    assert guess is not None
    assert guess["f"].item() == pytest.approx(frequency, rel=1e-12, abs=0)

    # Phases differing by an integer number of turns are equivalent.
    phase_error = (guess["phi"].item() - phase + 0.5) % 1.0 - 0.5
    assert phase_error == pytest.approx(0.0, abs=1e-10)


def test_damped_oscillation_guess_nonuniform():
    """The FFT initializer does not support nonuniform spacing"""
    time = np.array([0.0, 1.0, 2.2, 3.0, 4.0])
    data = xr.DataArray(
        np.cos(2 * np.pi * 0.2 * time),
        coords={"time": time},
    )

    assert DampedOscillationAnalysis.guess(data, coords="time") is None


def test_damped_oscillation_analysis_ramsey_good_snr():
    """Fit real Ramsey data with both zero and nonzero starting times."""
    ds, dim, units = open_test_dataset(
        "ramsey-good_snr-RX4_QD20260915022",
        dim="idle_time",
    )
    assert units == "ns"
    data = ds.Q22.assign_attrs(dataset_id=ds.source)

    results = []
    for first_sample in (0, 3):
        trace = data.isel({dim: slice(first_sample, None)})
        result = DampedOscillationAnalysis.run(trace, coords=dim)

        assert result.success.item()
        projected = result.intermediate_results.preprocessed_data
        fitted = DampedOscillationAnalysis.func(projected[dim], **result.fit_params)
        residual = fitted - projected
        normalized_rms = np.sqrt((residual**2).mean() / projected.var())

        # Both windows give about 0.13; allow some variation in the fit.
        assert normalized_rms.item() < 0.2
        results.append(result)

    full, cropped = results

    expected = ds.expected_fit_result["Q22"]
    assert full.params.f.item() * 1e9 == pytest.approx(
        expected["ramsey_frequency"], rel=0.01, abs=0
    )
    assert full.params.tau.item() * 1e-9 == pytest.approx(
        expected["t2_star"], rel=0.1, abs=0
    )
    assert cropped.params.f.item() == pytest.approx(
        full.params.f.item(), rel=1e-3, abs=0
    )
    assert cropped.params.tau.item() == pytest.approx(
        full.params.tau.item(), rel=0.05, abs=0
    )


@pytest.mark.xfail(
    reason="Default fitting is unreliable for cropped Q06 data in ns",
    strict=True,
)
def test_damped_oscillation_analysis_ramsey_cut_off():
    """Fit a short observation window using the default optimizer settings."""
    ds, dim, units = open_test_dataset(
        "ramsey-good_snr_cut_off-RX4_QD20260915022",
        dim="idle_time",
    )
    assert units == "ns"
    data = ds.Q06.assign_attrs(dataset_id=ds.source)

    trace = data.isel({dim: slice(3, None)})
    result = DampedOscillationAnalysis.run(trace, coords=dim)

    assert result.success.item()

    projected = result.intermediate_results.preprocessed_data
    fitted = DampedOscillationAnalysis.func(projected[dim], **result.fit_params)
    residual = fitted - projected
    normalized_rms = np.sqrt((residual**2).mean() / projected.var())

    assert normalized_rms.item() < 0.2

    # The short window provides a frequency reference, but no decay-time reference.
    expected = ds.expected_fit_result["Q06"]
    assert result.params.f.item() * 1e9 == pytest.approx(
        expected["ramsey_frequency"], rel=0.01, abs=0
    )


def test_damped_oscillation_analysis_ramsey_cut_off_with_jac_scaling():
    """
    Check that jacobian scaling gives good fit even for the short observation
    case. This test may be removed if
    test_damped_oscillation_analysis_ramsey_cut_off no longer xfails.
    """
    ds, dim, units = open_test_dataset(
        "ramsey-good_snr_cut_off-RX4_QD20260915022",
        dim="idle_time",
    )
    assert units == "ns"
    data = ds.Q06.assign_attrs(dataset_id=ds.source)

    trace = data.isel({dim: slice(3, None)})
    result = DampedOscillationAnalysis.run(
        trace, coords=dim, curvefit_kwargs={"kwargs": {"x_scale": "jac"}}
    )

    assert result.success.item()

    projected = result.intermediate_results.preprocessed_data
    fitted = DampedOscillationAnalysis.func(projected[dim], **result.fit_params)
    residual = fitted - projected
    normalized_rms = np.sqrt((residual**2).mean() / projected.var())

    assert normalized_rms.item() < 0.2

    # The short window provides a frequency reference, but no decay-time reference.
    expected = ds.expected_fit_result["Q06"]
    assert result.params.f.item() * 1e9 == pytest.approx(
        expected["ramsey_frequency"], rel=0.01, abs=0
    )


def test_damped_oscillation_analysis_ramsey_batch_with_missing_trace():
    """Select FFT peaks per trace without an all-NaN neighbor aborting the fit."""
    good, _, good_units = open_test_dataset(
        "ramsey-good_snr-RX4_QD20260915022",
        dim="idle_time",
    )
    cut_off, _, cut_off_units = open_test_dataset(
        "ramsey-good_snr_cut_off-RX4_QD20260915022",
        dim="idle_time",
    )
    assert good_units == cut_off_units == "ns"

    data = (
        xr.concat(
            [
                good.Q22.isel(idle_time=slice(0, 101)),
                cut_off.Q06,
                xr.full_like(cut_off.Q06, np.nan),
            ],
            dim=xr.IndexVariable("trace", ["good", "cut_off", "missing"]),
        )
        .transpose("idle_time", "trace")
        .assign_attrs(dataset_id="test-ramsey-batch")
    )
    result = DampedOscillationAnalysis.run(data, coords="idle_time")

    assert result.success.dims == ("trace",)
    assert result.success.trace.values.tolist() == ["good", "cut_off", "missing"]
    assert result.success.values.tolist() == [True, True, False]
    assert result.params.sel(trace="missing").to_array().isnull().all()

    valid = result.params.sel(trace=["good", "cut_off"])
    expected_frequencies = [
        good.expected_fit_result["Q22"]["ramsey_frequency"],
        cut_off.expected_fit_result["Q06"]["ramsey_frequency"],
    ]
    assert (valid.f * 1e9).values.tolist() == pytest.approx(
        expected_frequencies, rel=0.01, abs=0)

    projected = result.intermediate_results.preprocessed_data.sel(
        trace=["good", "cut_off"]
    )
    fitted = DampedOscillationAnalysis.func(
        projected.idle_time,
        **result.fit_params.sel(trace=["good", "cut_off"]),
    )
    normalized_rms = np.sqrt(
        ((fitted - projected) ** 2).mean("idle_time") / projected.var("idle_time")
    )
    assert (normalized_rms < 0.2).all()
