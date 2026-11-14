"""Synthetic, acquisition-independent validation of complex resonator analysis."""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from sqe_analysis.analysis import NotchResonatorAnalysis, ReflectionResonatorAnalysis
from sqe_analysis.analysis_base import CurvefitAnalysis
from sqe_analysis.result import CurvefitAnalysisResult


def sample(cls, ratio=0.6, seed=12, noise=2e-4):
    frequency = np.linspace(7.998e9, 8.002e9, 301)
    truth = {
        "fr": 8.0001e9,
        "Ql": 40000.0,
        "Qc": 40000.0 / ratio,
        "amplitude": 1.3,
        "phase": 0.4,
        "delay": 65e-9,
        "phi": 0.12,
        "reference_frequency": 8e9,
    }
    # Independent model expression guards against a factor-of-two regression.
    factor = 2 if cls is ReflectionResonatorAnalysis else 1
    d = 2j * truth["Ql"] * (frequency - truth["fr"]) / truth["fr"]
    z = truth["amplitude"] * np.exp(
        1j * (truth["phase"] - 2 * np.pi * (frequency - 8e9) * truth["delay"])
    )
    z *= 1 - factor * ratio * (1 + 1j * np.tan(truth["phi"])) / (1 + d)
    rng = np.random.default_rng(seed)
    z += noise * (rng.normal(size=z.size) + 1j * rng.normal(size=z.size))
    return xr.DataArray(
        z,
        dims="frequency",
        coords={"frequency": frequency},
        attrs={"dataset_id": "synthetic-resonator"},
        name="response",
    ), truth


@pytest.mark.parametrize("cls", [ReflectionResonatorAnalysis, NotchResonatorAnalysis])
@pytest.mark.parametrize("ratio", [0.08, 0.8])
def test_recovery_and_result_protocol(cls, ratio):
    data, truth = sample(cls, ratio)
    result = cls.run(data, coords="frequency")
    assert issubclass(cls, CurvefitAnalysis)
    assert isinstance(result, CurvefitAnalysisResult)
    assert result.success.item()
    assert result.source_dataset_id == "synthetic-resonator"
    assert result.params.fr.item() == pytest.approx(truth["fr"], abs=500)
    assert result.params.Qc.item() == pytest.approx(truth["Qc"], rel=0.02)
    assert result.params.Ql.item() == pytest.approx(truth["Ql"], rel=0.02)
    assert result.params.phi.item() == pytest.approx(truth["phi"], abs=0.01)
    assert result.params.delay.item() == pytest.approx(truth["delay"], abs=2e-9)
    assert result.params.Qi.item() == pytest.approx(
        1 / (1 / truth["Ql"] - 1 / truth["Qc"]), rel=0.03
    )
    prediction = cls.func(data.frequency, **result.fit_params)
    assert_allclose(prediction, result.intermediate_results.fitted_data)
    assert np.isfinite(result.params_std.Qc).all()
    assert (result.params_std.Qc > 0).all()
    assert result.debug_results.fit_covariance.shape == (7, 7)


def test_vectorized_guesses_failed_trace_and_transposed_input():
    data, _ = sample(ReflectionResonatorAnalysis)
    stacked = xr.concat(
        [data, data * 1.1, data * np.nan],
        dim=xr.IndexVariable("power", [-90, -80, -70]),
    )
    result = ReflectionResonatorAnalysis.run(
        stacked.transpose("frequency", "power"),
        coords="frequency",
        guess={
            "amplitude": xr.DataArray(
                [1.3, 1.43, 1.3], dims="power", coords={"power": [-90, -80, -70]}
            )
        },
    )
    assert result.params.fr.dims == ("power",)
    np.testing.assert_array_equal(result.success, [True, True, False])
    assert np.isnan(result.fit_params.fr.sel(power=-70))
    assert result.params.amplitude.sel(power=-80) / result.params.amplitude.sel(
        power=-90
    ) == pytest.approx(1.1, rel=0.001)
    assert_allclose(
        ReflectionResonatorAnalysis.func(stacked.frequency, **result.fit_params),
        result.intermediate_results.fitted_data,
    )


def test_missing_samples_and_raise_mode():
    data, _ = sample(NotchResonatorAnalysis)
    data.values[10:13] = np.nan + 1j * np.nan
    assert NotchResonatorAnalysis.run(data).success
    assert not NotchResonatorAnalysis.run(
        data, curvefit_kwargs={"skipna": False}
    ).success
    with pytest.raises(ValueError, match="finite"):
        NotchResonatorAnalysis.run(
            data, curvefit_kwargs={"skipna": False, "errors": "raise"}
        )


def test_malformed_inputs_and_options():
    data, _ = sample(NotchResonatorAnalysis)
    for invalid in [
        data.real,
        data.isel(frequency=slice(None, None, -1)),
        data.assign_coords(frequency=np.zeros(301)),
    ]:
        with pytest.raises(ValueError):
            NotchResonatorAnalysis.run(invalid)
    with pytest.raises(ValueError):
        NotchResonatorAnalysis.run(data, guess={"Qc": -2})
    with pytest.raises(ValueError):
        NotchResonatorAnalysis.run(data, bounds={"phi": (-2, 2)})
    with pytest.raises(ValueError):
        NotchResonatorAnalysis.run(data, curvefit_kwargs={"reduce_dims": ["power"]})
    with pytest.raises(ValueError):
        NotchResonatorAnalysis.run(data, guess={"typo": 1})


@pytest.mark.parametrize("cls", [ReflectionResonatorAnalysis, NotchResonatorAnalysis])
def test_flat_signal_rejected(cls):
    data, _ = sample(cls)
    data[:] = 1 + 0j
    assert not cls.run(data).success


@pytest.mark.parametrize("cls", [ReflectionResonatorAnalysis, NotchResonatorAnalysis])
def test_delay_only_and_noisy_background_rejected(cls):
    data, _ = sample(cls)
    f = data.frequency.values
    rng = np.random.default_rng(123)
    data[:] = np.exp(-2j * np.pi * (f - 8e9) * 1e-6)
    assert not cls.run(data).success
    data[:] += 0.002 * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
    assert not cls.run(data).success


def test_covariance_units_and_noise_scaling():
    cls = ReflectionResonatorAnalysis
    low, _ = sample(cls, noise=1e-4)
    high, _ = sample(cls, noise=4e-4)
    a, b = cls.run(low), cls.run(high)
    assert float(b.params_std.fr / a.params_std.fr) == pytest.approx(4, rel=0.05)
    assert float(b.params_std.Qc / a.params_std.Qc) == pytest.approx(4, rel=0.05)


def test_multidimensional_bounds_and_manual_guesses():
    data, truth = sample(NotchResonatorAnalysis)
    power = xr.IndexVariable("power", [-90, -80])
    qubit = xr.IndexVariable("qubit", ["a", "b"])
    stacked = xr.concat([xr.concat([data, data], dim=power)] * 2, dim=qubit)
    result = NotchResonatorAnalysis.run(
        stacked,
        "frequency",
        guess={"fr": truth["fr"], "Ql": truth["Ql"], "Qc": truth["Qc"]},
        bounds={
            "fr": (
                xr.DataArray([8e9, 8e9], dims="power", coords={"power": power}),
                8.0002e9,
            )
        },
    )
    assert result.success.all()
    assert result.params.fr.dims == ("qubit", "power")
    assert float(result.params.fr.sel(qubit="b", power=-80)) == pytest.approx(
        truth["fr"], abs=500
    )


def test_unresolved_resonance_is_not_successful():
    cls = ReflectionResonatorAnalysis
    data, truth = sample(cls, noise=0)
    truth.update(Ql=2e6, Qc=3e6)
    data[:] = cls.func(data.frequency, **truth)
    result = cls.run(data, guess={key: truth[key] for key in ("fr", "Ql", "Qc")})
    assert not result.success


def test_invalid_configuration_is_not_ignored():
    data, _ = sample(NotchResonatorAnalysis)
    for kwargs in [
        {"guess": {"fr": 1e9}},
        {"bounds": {"fr": (1e9, 2e9)}},
        {"curvefit_kwargs": {"kwargs": {"max_nfev": 0}}},
        {"guess": {"amplitude": xr.DataArray([1, 2], dims="unrelated")}},
    ]:
        with pytest.raises(ValueError):
            NotchResonatorAnalysis.run(data, **kwargs)


def test_quality_rejection_retains_fitted_parameters():
    data, _ = sample(NotchResonatorAnalysis)
    result = NotchResonatorAnalysis.run(data, max_relative_std=1e-8)
    assert not result.success
    assert np.isfinite(result.params.fr)
    assert np.isfinite(result.params_std.Qc)
