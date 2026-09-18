"""
Tests for DampedOscillationAnalysis
"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose, assert_identical

from sqe_analysis.analysis import DampedOscillationAnalysis
from sqe_analysis.signal_processing import project_complex


@pytest.mark.parametrize("automatic_f", [False, True])
@pytest.mark.parametrize("time_unit, time_scale", [("s", 1.0), ("us", 1e6)])
def test_damped_oscillation_analysis_basic(time_unit, time_scale, automatic_f):
    """Recover a noiseless oscillation with manual or automatic guesses."""
    time = np.linspace(0, 40e-6, 401) * time_scale
    tau = 12e-6 * time_scale
    f = 0.4e6 / time_scale

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time / tau) * np.cos(2 * np.pi * (f * time + 0.4)),
        coords=[("time", time)],
        attrs={"dataset_id": "test"},
    )
    data.time.attrs["units"] = time_unit
    original_data = data.copy(deep=True)

    guess = {"a": 0.7, "b": 0.1, "tau": tau * 0.8, "phi": 0.3}
    if not automatic_f:
        guess["f"] = f * 1.01

    result = DampedOscillationAnalysis.run(
        data,
        coords="time",
        guess=guess,
    )

    if automatic_f:
        frequency_step = 1 / (time.size * (time[1] - time[0]))
        assert result.fit_params_guess.f.item() == pytest.approx(f, abs=frequency_step)
    else:
        assert result.fit_params_guess.f.item() == pytest.approx(guess["f"])

    assert result.success.all()
    assert result.params.a.item() == pytest.approx(0.8)
    assert result.params.b.item() == pytest.approx(0.2)
    assert result.params.tau.item() == pytest.approx(tau, rel=1e-6, abs=0)
    assert result.params.f.item() == pytest.approx(f)
    assert result.params.phi.item() == pytest.approx(0.4)

    assert_allclose(
        DampedOscillationAnalysis.func(data.time, **result.fit_params),
        data,
        rtol=1e-6,
        atol=1e-8,
    )
    assert_identical(data, original_data)


def test_damped_oscillation_analysis_nonuniform_time_with_manual_guess():
    """Fit nonuniformly sampled data using explicitly supplied guesses."""
    time = np.linspace(0, 40e-6, 401)
    time[1::2] += 20e-9

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time / 12e-6) * np.cos(2 * np.pi * (400e3 * time + 0.4)),
        coords=[("time", time)],
        attrs={"dataset_id": "test"},
    )
    data.time.attrs["units"] = "s"
    original_data = data.copy(deep=True)

    assert DampedOscillationAnalysis.guess(data, coords="time") is None

    result = DampedOscillationAnalysis.run(
        data,
        coords="time",
        guess={"a": 0.7, "b": 0.1, "tau": 10e-6, "f": 404e3, "phi": 0.3},
    )

    assert result.success.all()
    assert result.params.tau.item() == pytest.approx(12e-6, rel=1e-6, abs=0)
    assert result.params.f.item() == pytest.approx(400e3)
    assert_allclose(
        DampedOscillationAnalysis.func(data.time, **result.fit_params),
        data,
        rtol=1e-6,
        atol=1e-8,
    )
    assert_identical(data, original_data)


@pytest.mark.parametrize("manual_ab", [False, True])
@pytest.mark.parametrize("time_unit, time_scale", [("s", 1.0), ("us", 1e6)])
def test_damped_oscillation_analysis_amplitude_offset_guess(
    time_unit, time_scale, manual_ab
):
    """Recover parameters with automatic or overridden amplitude and offset."""
    time = np.linspace(0, 40e-6, 401) * time_scale
    tau = 12e-6 * time_scale
    f = 0.4e6 / time_scale

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time / tau) * np.cos(2 * np.pi * (f * time + 0.4)),
        coords=[("time", time)],
        attrs={"dataset_id": "test"},
    )
    data.time.attrs["units"] = time_unit
    original_data = data.copy(deep=True)

    guess = {"tau": tau * 0.8, "phi": 0.3}
    if manual_ab:
        guess.update({"a": 0.7, "b": 0.1})

    result = DampedOscillationAnalysis.run(
        data,
        coords="time",
        guess=guess,
    )

    assert set(result.fit_params_guess.data_vars) == {"a", "b", "tau", "f", "phi"}

    if manual_ab:
        assert result.fit_params_guess.a.item() == pytest.approx(0.7)
        assert result.fit_params_guess.b.item() == pytest.approx(0.1)

    assert result.success.all()
    assert result.params.a.item() == pytest.approx(0.8)
    assert result.params.b.item() == pytest.approx(0.2)
    assert result.params.tau.item() == pytest.approx(tau, rel=1e-6, abs=0)
    assert result.params.f.item() == pytest.approx(f)
    assert result.params.phi.item() == pytest.approx(0.4)

    assert_allclose(
        DampedOscillationAnalysis.func(data.time, **result.fit_params),
        data,
        rtol=1e-6,
        atol=1e-8,
    )
    assert_identical(data, original_data)


@pytest.mark.parametrize(
    "time_unit, time_scale",
    [("s", 1.0), ("us", 1e6), ("ns", 1e9)],
)
def test_damped_oscillation_analysis_without_guess(time_unit, time_scale):
    """Recover the same physical paramters withou explicit initial guessses."""
    time_s = np.linspace(0, 40e-6, 401)

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time_s / 12e-6) * np.cos(2 * np.pi * 400e3 * time_s + 0.4),
        coords=[("idle_time", time_s * time_scale)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = time_unit

    result = DampedOscillationAnalysis.run(
        data,
        coords="idle_time",
    )

    assert result.success.all()

    tau_s = result.params.tau.item() / time_scale
    frequency_hz = result.params.f.item() * time_scale

    assert tau_s == pytest.approx(12e-6, rel=1e-5, abs=0)
    assert frequency_hz == pytest.approx(400e3, rel=1e-5, abs=0)

    assert_allclose(
        DampedOscillationAnalysis.func(data.idle_time, **result.fit_params),
        data,
        rtol=1e-5,
        atol=1e-7,
    )

    assert result.fit_params_guess is not None
    assert set(result.fit_params_guess.data_vars) == {
        "a",
        "b",
        "tau",
        "f",
        "phi",
    }


@pytest.mark.parametrize(
    "time_unit, time_scale",
    [("s", 1.0), ("us", 1e6), ("ns", 1e9)],
)
def test_damped_oscillation_analysis_positive_tau_bounds(time_unit, time_scale):
    """Recover a bounded fit and reject an out-of-bounds initial decay time."""
    time_s = np.linspace(0, 40e-6, 401)

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time_s / 12e-6) * np.cos(2 * np.pi * 400e3 * time_s + 0.4),
        coords=[("idle_time", time_s * time_scale)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = time_unit

    curvefit_kwargs = {
        "bounds": {"tau": (0, np.inf)},
        "kwargs": {"x_scale": "jac"},
    }

    result = DampedOscillationAnalysis.run(
        data,
        coords="idle_time",
        curvefit_kwargs=curvefit_kwargs,
    )

    assert result.success.all()

    tau_s = result.params.tau.item() / time_scale
    frequency_hz = result.params.f.item() * time_scale

    assert np.isfinite(tau_s)
    assert tau_s > 0
    assert tau_s == pytest.approx(12e-6, rel=1e-5, abs=0)
    assert frequency_hz == pytest.approx(400e3, rel=1e-5, abs=0)

    assert_allclose(
        DampedOscillationAnalysis.func(data.idle_time, **result.fit_params),
        data,
        rtol=1e-5,
        atol=1e-7,
    )

    with pytest.raises(ValueError, match="bounds"):
        DampedOscillationAnalysis.run(
            data,
            coords="idle_time",
            guess={"tau": -12e-6 * time_scale},
            curvefit_kwargs=curvefit_kwargs,
        )


def test_damped_oscillation_analysis_rejects_negative_tau_guess():
    """Reject a negative initial decay time without explicit bounds."""
    time = np.linspace(0, 40, 401)

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time / 12) * np.cos(2 * np.pi * 0.4 * time + 0.4),
        coords=[("idle_time", time)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = "us"

    with pytest.raises(ValueError):
        DampedOscillationAnalysis.run(
            data,
            coords="idle_time",
            guess={"tau": -12.0},
        )


@pytest.mark.parametrize("iq_rotation", [0.4, np.pi / 2, 2.0])
def test_damped_oscillation_analysis_complex_iq(iq_rotation):
    """Recover decay time and frequency from rotated complex readout IQ."""
    time = np.linspace(0, 40, 401)
    signal = 0.2 + 0.8 * np.exp(-time / 12) * np.cos(2 * np.pi * 0.4 * time + 0.4)

    data = xr.DataArray(
        (1.1 - 0.6j) + np.exp(1j * iq_rotation) * signal,
        coords=[("idle_time", time)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    projected = DampedOscillationAnalysis.preprocess(
        data,
        coords="idle_time",
    )
    assert projected is not None
    assert not np.iscomplexobj(projected)
    assert_allclose(projected, project_complex(data, dim="idle_time"))

    result = DampedOscillationAnalysis.run(data, coords="idle_time")

    assert result.success.all()
    assert result.params.tau.item() == pytest.approx(12, rel=1e-5, abs=0)
    assert result.params.f.item() == pytest.approx(0.4, rel=1e-5, abs=0)

    assert result.intermediate_results is not None
    recorded = result.intermediate_results.preprocessed_data
    assert_allclose(recorded, projected.rename("preprocessed_data"))

    assert_allclose(
        DampedOscillationAnalysis.func(recorded.idle_time, **result.fit_params),
        recorded,
        rtol=1e-5,
        atol=1e-7,
    )
    assert_identical(data, original_data)


@pytest.mark.parametrize("time_first", [False, True])
def test_damped_oscillation_analysis_multiple_traces(time_first):
    """Recover separate parameters while preserving trace coordinates."""
    time_values = np.linspace(0, 40, 401)
    time = xr.DataArray(
        time_values,
        coords=[("idle_time", time_values)],
    )
    expected = xr.Dataset(
        {
            "tau": ("trace", [8.0, 18.0]),
            "f": ("trace", [0.4, 0.7]),
        },
        coords={"trace": ["short", "long"]},
    )

    data = 0.2 + 0.8 * np.exp(-time / expected.tau) * np.cos(
        2 * np.pi * expected.f * time + 0.4
    )
    if time_first:
        data = data.transpose("idle_time", "trace")
    else:
        data = data.transpose("trace", "idle_time")

    data.attrs["dataset_id"] = "test"
    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    guesses = DampedOscillationAnalysis.guess(data, coords="idle_time")

    assert guesses is not None
    assert set(guesses) == {"a", "b", "tau", "f", "phi"}
    assert isinstance(guesses["f"], xr.DataArray)
    assert guesses["f"].dims == ("trace",)
    assert_identical(guesses["f"].trace, expected.trace)

    frequency_step = 1 / (time_values.size * (time_values[1] - time_values[0]))
    assert_allclose(
        guesses["f"].rename("f"),
        expected.f,
        rtol=0,
        atol=frequency_step,
    )

    result = DampedOscillationAnalysis.run(data, coords="idle_time")

    assert result.success.dims == ("trace",)
    assert_identical(result.success.trace, expected.trace)
    assert result.success.all()
    assert_allclose(
        result.params[["tau", "f"]],
        expected,
        rtol=1e-5,
        atol=0,
    )

    fitted = DampedOscillationAnalysis.func(
        data.idle_time,
        **result.fit_params,
    ).transpose(*data.dims)

    assert_allclose(fitted, data, rtol=1e-5, atol=1e-7)
    assert_identical(data, original_data)


@pytest.mark.parametrize("time_first", [False, True])
def test_damped_oscillation_analysis_complex_iq_multiple_traces(time_first):
    """Recover separate decay parameters from differently rotated IQ traces."""
    time_values = np.linspace(0, 40, 401)
    time = xr.DataArray(
        time_values,
        coords=[("idle_time", time_values)],
    )
    expected = xr.Dataset(
        {
            "tau": ("trace", [8.0, 12.0, 18.0]),
            "f": ("trace", [0.4, 0.55, 0.7]),
            "iq_rotation": ("trace", [0.4, np.pi / 2, 2.0]),
            "iq_offset": (
                "trace",
                [1.1 - 0.6j, -0.4 + 0.9j, 0.7 + 0.3j],
            ),
        },
        coords={"trace": ["short", "middle", "long"]},
    )

    # Generate real oscillations, then rotate and offset each IQ trace.
    signal = 0.2 + 0.8 * np.exp(-time / expected.tau) * np.cos(
        2 * np.pi * expected.f * time + 0.4
    )
    data = expected.iq_offset + np.exp(1j * expected.iq_rotation) * signal

    if time_first:
        data = data.transpose("idle_time", "trace")
    else:
        data = data.transpose("trace", "idle_time")

    data.attrs["dataset_id"] = "test"
    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    # Run the full analysis without manual projection or initial guesses.
    result = DampedOscillationAnalysis.run(data, coords="idle_time")

    assert result.success.dims == ("trace",)
    assert_identical(result.success.trace, expected.trace)
    assert result.success.all()
    assert_allclose(
        result.params[["tau", "f"]],
        expected[["tau", "f"]],
        rtol=1e-5,
        atol=0,
    )

    # Projection should recover the centered signal up to an overall sign
    # for each trace.
    assert result.intermediate_results is not None
    recorded = result.intermediate_results.preprocessed_data.transpose(*data.dims)
    assert not np.iscomplexobj(recorded)

    expected_projection = (
        (signal - signal.mean("idle_time"))
        .transpose(*data.dims)
        .rename("preprocessed_data")
    )
    assert_allclose(
        abs(recorded),
        abs(expected_projection),
        rtol=1e-7,
        atol=1e-12,
    )

    # The fitted curve should reproduce the actual projected signal.
    fitted = DampedOscillationAnalysis.func(
        recorded.idle_time,
        **result.fit_params,
    ).transpose(*data.dims)
    assert_allclose(fitted, recorded, rtol=1e-5, atol=1e-7)

    assert_identical(data, original_data)


@pytest.mark.parametrize(
    "bounded_frequency",
    [False, True],
    ids=["default", "bounded"],
)
@pytest.mark.parametrize("time_first", [False, True])
def test_damped_oscillation_analysis_constant_trace_does_not_affect_signal(
    time_first,
    bounded_frequency,
):
    """Reject a constant trace while recovering a neighboring oscillation."""
    time = np.linspace(0, 40, 401)
    signal = 0.2 + 0.8 * np.exp(-time / 12) * np.cos(2 * np.pi * 0.4 * time + 0.4)

    data = xr.DataArray(
        np.stack([signal, np.full_like(time, 0.2)]),
        coords=[
            ("trace", ["signal", "constant"]),
            ("idle_time", time),
        ],
        attrs={"dataset_id": "test"},
    )
    if time_first:
        data = data.transpose("idle_time", "trace")

    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    curvefit_kwargs = {}
    if bounded_frequency:
        curvefit_kwargs["bounds"] = {"f": (0.3, 0.5)}

    # Analyze both traces together, without manual initial guesses.
    result = DampedOscillationAnalysis.run(
        data,
        coords="idle_time",
        curvefit_kwargs=curvefit_kwargs,
    )

    # Automatic guesses must remain available for the valid trace.
    assert result.fit_params_guess is not None

    initial_frequency = result.fit_params_guess.f
    assert initial_frequency.dims == ("trace",)
    assert_identical(initial_frequency.trace, data.trace)

    frequency_step = 1 / (time.size * (time[1] - time[0]))
    assert initial_frequency.sel(trace="signal").item() == pytest.approx(
        0.4,
        rel=0,
        abs=frequency_step,
    )

    assert result.success.dims == ("trace",)
    assert_identical(result.success.trace, data.trace)

    # A constant trace cannot determine a decay time or frequency.
    assert not result.success.sel(trace="constant").item()

    for params in (result.params, result.fit_params):
        invalid = params[["tau", "f"]].sel(trace="constant")
        assert invalid.to_array().isnull().all()

    # The valid trace must still recover its known parameters.
    assert result.success.sel(trace="signal").item()

    valid_params = result.params.sel(trace="signal", drop=True)
    assert valid_params.tau.item() == pytest.approx(12.0, rel=1e-5, abs=0)
    assert valid_params.f.item() == pytest.approx(0.4, rel=1e-5, abs=0)

    valid_data = data.sel(trace="signal", drop=True)
    fitted = DampedOscillationAnalysis.func(
        valid_data.idle_time,
        **result.fit_params.sel(trace="signal", drop=True),
    )
    assert_allclose(fitted, valid_data, rtol=1e-5, atol=1e-7)

    assert_identical(data, original_data)

    if bounded_frequency:
        assert curvefit_kwargs == {"bounds": {"f": (0.3, 0.5)}}

        with pytest.raises(ValueError, match="bounds"):
            DampedOscillationAnalysis.run(
                data,
                coords="idle_time",
                guess={"f": 0.2},
                curvefit_kwargs=curvefit_kwargs,
            )
    else:
        assert curvefit_kwargs == {}


@pytest.mark.parametrize("time_first", [False, True])
def test_damped_oscillation_analysis_all_nan_trace_does_not_affect_signal(
    time_first,
):
    """Preserve automatic fitting beside a completely missing trace."""
    time = np.linspace(0, 40, 401)
    signal = 0.2 + 0.8 * np.exp(-time / 12) * np.cos(2 * np.pi * 0.4 * time + 0.4)

    data = xr.DataArray(
        np.stack([signal, np.full_like(time, np.nan)]),
        coords=[
            ("trace", ["signal", "missing"]),
            ("idle_time", time),
        ],
        attrs={"dataset_id": "test"},
    )
    if time_first:
        data = data.transpose("idle_time", "trace")

    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    result = DampedOscillationAnalysis.run(data, coords="idle_time")

    # Missing data must not disable automatic guesses for the valid trace.
    assert result.fit_params_guess is not None

    initial_frequency = result.fit_params_guess.f
    assert initial_frequency.dims == ("trace",)
    assert_identical(initial_frequency.trace, data.trace)

    frequency_step = 1 / (time.size * (time[1] - time[0]))
    assert initial_frequency.sel(trace="signal").item() == pytest.approx(
        0.4,
        rel=0,
        abs=frequency_step,
    )

    # Preserve trace labels and report the missing trace as unsuccessful.
    assert result.success.dims == ("trace",)
    assert_identical(result.success.trace, data.trace)
    assert not result.success.sel(trace="missing").item()

    for params in (result.params, result.fit_params):
        missing = params[["tau", "f"]].sel(trace="missing")
        assert missing.to_array().isnull().all()

    # The valid trace must still recover its known parameters and waveform.
    assert result.success.sel(trace="signal").item()

    valid_params = result.params.sel(trace="signal", drop=True)
    assert valid_params.tau.item() == pytest.approx(
        12.0,
        rel=1e-5,
        abs=0,
    )
    assert valid_params.f.item() == pytest.approx(
        0.4,
        rel=1e-5,
        abs=0,
    )

    valid_data = data.sel(trace="signal", drop=True)
    fitted = DampedOscillationAnalysis.func(
        valid_data.idle_time,
        **result.fit_params.sel(trace="signal", drop=True),
    )
    assert_allclose(fitted, valid_data, rtol=1e-5, atol=1e-7)

    assert_identical(data, original_data)


@pytest.mark.parametrize(
    "invalid_time",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "positive_inf", "negative_inf"],
)
def test_damped_oscillation_analysis_rejects_nonfinite_time(invalid_time):
    """Reject nonfinite time coordinates even with manual guesses."""
    time = np.linspace(0, 40, 401)
    signal = 0.2 + 0.8 * np.exp(-time / 12) * np.cos(2 * np.pi * 0.4 * time + 0.4)

    invalid_coordinate = time.copy()
    invalid_coordinate[100] = invalid_time

    data = xr.DataArray(
        signal,
        coords=[("idle_time", invalid_coordinate)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    with pytest.raises(
        ValueError,
        match="Time coordinates must contain only finite values",
    ):
        DampedOscillationAnalysis.run(
            data,
            coords="idle_time",
            guess={
                "a": 0.7,
                "b": 0.1,
                "tau": 10.0,
                "f": 0.404,
                "phi": 0.3,
            },
        )

    assert_identical(data, original_data)


@pytest.mark.parametrize(
    "time_order",
    ["duplicate", "descending", "local_reversal"],
)
@pytest.mark.parametrize("guess_mode", ["none", "partial", "complete"])
def test_damped_oscillation_analysis_nonincreasing_time(
    time_order,
    guess_mode,
):
    """Require complete manual guesses for non-increasing time."""
    time = np.linspace(0, 40, 401)

    if time_order == "duplicate":
        time[100] = time[99]
    elif time_order == "descending":
        time = time[::-1]
    else:
        time[[100, 101]] = time[[101, 100]]

    signal = 0.2 + 0.8 * np.exp(-time / 12) * np.cos(2 * np.pi * 0.4 * time + 0.4)

    data = xr.DataArray(
        signal,
        coords=[("idle_time", time)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = "us"
    original_data = data.copy(deep=True)

    if guess_mode == "none":
        guess = None
    elif guess_mode == "partial":
        guess = {"f": 0.404}
    else:
        guess = {
            "a": 0.7,
            "b": 0.1,
            "tau": 10.0,
            "f": 0.404,
            "phi": 0.3,
        }

    original_guess = None if guess is None else dict(guess)

    assert DampedOscillationAnalysis.guess(data, coords="idle_time") is None

    if guess_mode == "complete":
        result = DampedOscillationAnalysis.run(
            data,
            coords="idle_time",
            guess=guess,
        )

        assert result.success.item()
        assert result.params.tau.item() == pytest.approx(
            12.0,
            rel=1e-5,
            abs=0,
        )
        assert result.params.f.item() == pytest.approx(
            0.4,
            rel=1e-5,
            abs=0,
        )

        fitted = DampedOscillationAnalysis.func(
            data.idle_time,
            **result.fit_params,
        )
        assert_allclose(fitted, data, rtol=1e-5, atol=1e-7)
    else:
        with pytest.raises(
            ValueError,
            match="Non-increasing time coordinates require initial guesses",
        ):
            DampedOscillationAnalysis.run(
                data,
                coords="idle_time",
                guess=guess,
            )

    assert_identical(data, original_data)
    assert guess == original_guess


@pytest.mark.parametrize(
    "time_unit, time_scale",
    [("s", 1.0), ("us", 1e6), ("ns", 1e9)],
)
@pytest.mark.parametrize("guess_mode", ["none", "partial", "complete"])
def test_damped_oscillation_analysis_nonuniform_time_requires_complete_guess(
    time_unit,
    time_scale,
    guess_mode,
):
    """Require complete manual guesses for nonuniform time."""
    time_s = np.linspace(0, 40e-6, 401)
    time_s[1::2] += 20e-9

    signal = 0.2 + 0.8 * np.exp(-time_s / 12e-6) * np.cos(
        2 * np.pi * 400e3 * time_s + 0.4
    )

    data = xr.DataArray(
        signal,
        coords=[("idle_time", time_s * time_scale)],
        attrs={"dataset_id": "test"},
    )
    data.idle_time.attrs["units"] = time_unit
    original_data = data.copy(deep=True)

    if guess_mode == "none":
        guess = None
    elif guess_mode == "partial":
        guess = {"f": 404e3 / time_scale}
    else:
        guess = {
            "a": 0.7,
            "b": 0.1,
            "tau": 10e-6 * time_scale,
            "f": 404e3 / time_scale,
            "phi": 0.3,
        }

    original_guess = None if guess is None else dict(guess)

    assert DampedOscillationAnalysis.guess(data, coords="idle_time") is None

    if guess_mode == "complete":
        result = DampedOscillationAnalysis.run(
            data,
            coords="idle_time",
            guess=guess,
        )

        assert result.success.item()
        assert result.params.tau.item() / time_scale == pytest.approx(
            12e-6,
            rel=1e-5,
            abs=0,
        )
        assert result.params.f.item() * time_scale == pytest.approx(
            400e3,
            rel=1e-5,
            abs=0,
        )

        assert result.fit_params_guess is not None
        for name, value in guess.items():
            assert result.fit_params_guess[name].item() == pytest.approx(
                value,
                rel=1e-12,
                abs=0,
            )

        fitted = DampedOscillationAnalysis.func(
            data.idle_time,
            **result.fit_params,
        )
        assert_allclose(fitted, data, rtol=1e-5, atol=1e-7)
    else:
        with pytest.raises(
            ValueError,
            match="Nonuniform time coordinates require initial guesses",
        ):
            DampedOscillationAnalysis.run(
                data,
                coords="idle_time",
                guess=guess,
            )

    assert_identical(data, original_data)
    assert guess == original_guess
