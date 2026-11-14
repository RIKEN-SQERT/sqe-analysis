"""Independent synthetic signals for real peak, fringe, and oscillation models."""

import numpy as np
import pytest
import xarray as xr

from sqe_analysis.analysis import (
    DampedBeatingOscillationAnalysis,
    DoubleGaussianAnalysis,
    DoubleLorentzianAnalysis,
    ExponentialRegressionAnalysis,
    GaussianAnalysis,
    LinearAnalysis,
    LorentzianAnalysis,
    OscillationAnalysis,
    RamseyFringeAnalysis,
    StepExponentialAnalysis,
)


def array(x, y):
    return xr.DataArray(y, dims="x", coords={"x": x}, attrs={"dataset_id": "synthetic"})


@pytest.mark.parametrize(
    "model,signal,expected",
    [
        (LinearAnalysis, lambda x: 0.3 * x + 0.2, {"a": 0.3, "b": 0.2}),
        (
            LorentzianAnalysis,
            lambda x: 0.2 - 0.7 / (1 + ((x - 5) / 0.4) ** 2),
            {"c": 5, "sigma": 0.4},
        ),
        (
            DoubleGaussianAnalysis,
            lambda x: (
                0.1
                + np.exp(-0.5 * ((x - 3) / 0.5) ** 2)
                + 0.7 * np.exp(-0.5 * ((x - 8) / 0.8) ** 2)
            ),
            {"c1": 3, "c2": 8, "sigma1": 0.5, "sigma2": 0.8},
        ),
        (
            DoubleLorentzianAnalysis,
            lambda x: (
                0.1 - 1 / (1 + ((x - 3) / 0.3) ** 2) - 0.7 / (1 + ((x - 8) / 0.5) ** 2)
            ),
            {"c1": 3, "c2": 8, "sigma1": 0.3, "sigma2": 0.5},
        ),
        (
            OscillationAnalysis,
            lambda x: 0.2 + 0.7 * np.cos(2 * np.pi * (0.8 * x + 0.1)),
            {"f": 0.8},
        ),
        (
            DampedBeatingOscillationAnalysis,
            lambda x: (
                0.2
                + 0.7
                * np.exp(-x / 7)
                * (
                    np.cos(2 * np.pi * 0.8 * (x - 0.1))
                    + np.cos(2 * np.pi * 1.3 * (x - 0.1))
                )
            ),
            {"tau": 7, "f": 0.8, "s": 0.5},
        ),
        (
            RamseyFringeAnalysis,
            lambda x: (
                0.1
                + 0.7
                * (1 - 0.6 * np.cos(2 * np.pi * (1.7 * (x - 6) + 0.1)))
                * np.exp(-0.5 * ((x - 6) / 2) ** 2)
            ),
            {"c": 6, "sigma": 2, "f": 1.7, "contrast": 0.6},
        ),
    ],
)
def test_automatic_synthetic_fits(model, signal, expected):
    x = np.linspace(0, 12, 601)
    y = signal(x) + np.random.default_rng(42).normal(0, 0.002, len(x))
    result = model.run(array(x, y), coords="x")
    assert result.success
    for name, value in expected.items():
        assert float(result.params[name]) == pytest.approx(value, rel=0.03, abs=0.02)
        assert np.isfinite(result.params_std[name]) and result.params_std[name] >= 0
    np.testing.assert_allclose(
        model.func(array(x, y).x, **result.fit_params),
        result.intermediate_results.fitted_data,
    )
    assert result.source_dataset_id == "synthetic"


def test_linear_covariance_matches_closed_form_and_vectorization():
    x = np.linspace(-3, 7, 81)
    noise = np.random.default_rng(8).normal(0, 0.1, len(x))
    data = array(x, 2 * x + 1 + noise).expand_dims(trace=["one", "two"])
    result = LinearAnalysis.run(data, coords="x")
    design = np.column_stack([x, np.ones_like(x)])
    residual = data.sel(trace="one").values - design @ [
        float(result.params.a[0]),
        float(result.params.b[0]),
    ]
    covariance = np.linalg.inv(design.T @ design) * np.sum(residual**2) / (len(x) - 2)
    np.testing.assert_allclose(
        result.debug_results.covariance.sel(trace="one"), covariance, rtol=1e-5
    )
    np.testing.assert_allclose(
        result.params_std.a, np.sqrt(covariance[0, 0]), rtol=1e-5
    )
    assert result.params_std.a.dims == ("trace",)


def test_peak_models_preserve_real_baseline_and_nan_samples():
    x = np.linspace(-3, 3, 201)
    y = 3 + 1 / (1 + (x / 0.4) ** 2)
    y[20:25] = np.nan
    result = LorentzianAnalysis.run(
        array(x, y), coords="x", curvefit_kwargs={"skipna": True}
    )
    assert float(result.params.b) == pytest.approx(3, abs=1e-5)
    assert float(result.params.FWHM) == pytest.approx(0.8, abs=1e-5)
    np.testing.assert_allclose(result.params_std.FWHM, 2 * result.params_std.sigma)


def test_gaussian_uncertainty_is_propagated_to_fwhm():
    x = np.linspace(-3, 3, 101)
    y = np.exp(-0.5 * (x / 0.6) ** 2) + np.random.default_rng(2).normal(0, 0.01, len(x))
    result = GaussianAnalysis.run(array(x, y), coords="x")
    assert result.params_std is not None
    np.testing.assert_allclose(
        result.params_std.FWHM, result.params_std.sigma * 2 * np.sqrt(2 * np.log(2))
    )


def test_beating_second_frequency_uses_correlations():
    x = np.linspace(0, 15, 601)
    y = 0.2 + 0.7 * np.exp(-x / 7) * (
        np.cos(2 * np.pi * 0.8 * x) + np.cos(2 * np.pi * 1.3 * x)
    )
    y += np.random.default_rng(3).normal(0, 0.01, len(x))
    result = DampedBeatingOscillationAnalysis.run(array(x, y), coords="x")
    covariance = result.debug_results.covariance
    expected = (
        covariance.sel(cov_i="f", cov_j="f")
        + covariance.sel(cov_i="s", cov_j="s")
        + 2 * covariance.sel(cov_i="f", cov_j="s")
    )
    assert float(result.params.second_frequency) == pytest.approx(1.3, rel=0.01)
    np.testing.assert_allclose(result.params_std.second_frequency**2, expected)


def test_ramsey_extremum_is_not_assumed_to_be_envelope_center():
    x = np.linspace(0, 12, 601)
    y = 0.1 + 0.7 * (1 - 0.6 * np.cos(2 * np.pi * (1.7 * (x - 6) + 0.1))) * np.exp(
        -0.5 * ((x - 6) / 2) ** 2
    )
    result = RamseyFringeAnalysis.run(array(x, y), coords="x")
    dense_x = np.linspace(0, 12, 100001)
    dense_y = 0.1 + 0.7 * (
        1 - 0.6 * np.cos(2 * np.pi * (1.7 * (dense_x - 6) + 0.1))
    ) * np.exp(-0.5 * ((dense_x - 6) / 2) ** 2)
    assert float(result.params.extremum_location) == pytest.approx(
        dense_x[dense_y.argmax()], abs=0.003
    )
    assert np.isnan(result.params_std.extremum_location)


def test_nanosecond_peak_coordinates():
    x = np.linspace(-5e-9, 5e-9, 201)
    y = 0.2 + 1 / (1 + ((x - 1e-9) / 0.4e-9) ** 2)
    result = LorentzianAnalysis.run(array(x, y), coords="x")
    assert float(result.params.c) == pytest.approx(1e-9, rel=0.01)
    assert float(result.params.sigma) == pytest.approx(0.4e-9, rel=0.01)


def test_ghz_ramsey_frequency_axis_and_missing_trace_isolation():
    x = 5e9 + np.linspace(-6e6, 6e6, 601)
    y = 0.1 + 0.7 * (1 - 0.6 * np.cos(2 * np.pi * (1.7e-6 * (x - 5e9) + 0.1))) * np.exp(
        -0.5 * ((x - 5e9) / 2e6) ** 2
    )
    result = RamseyFringeAnalysis.run(array(x, y), coords="x")
    assert float(result.params.c) == pytest.approx(5e9, abs=1000)
    assert float(result.params.f) == pytest.approx(1.7e-6, rel=1e-4)
    batch = xr.concat([array(x, y), array(x, np.full_like(y, np.nan))], dim="trace")
    fitted = LorentzianAnalysis.run(batch, coords="x")
    assert not fitted.success.isel(trace=1)
    assert np.isnan(fitted.params_std.c.isel(trace=1))


@pytest.mark.parametrize("solver", [{"maxfev": 2000}, {"method": "lm", "maxfev": 2000}])
def test_linear_scipy_solver_options(solver):
    x = np.linspace(0, 3, 31)
    result = LinearAnalysis.run(
        array(x, 2 * x + 1), coords="x", curvefit_kwargs={"kwargs": solver}
    )
    assert float(result.params.a) == pytest.approx(2.0)


def test_step_profile_recovers_edge_and_conditional_errors():
    x = np.linspace(0, 12, 241)
    y = np.where(x >= 2, 0.1 + 0.8 * np.exp(-np.maximum(x - 2, 0) / 3), 0.0)
    y += np.random.default_rng(2).normal(0, 0.002, len(x))
    data = array(x, y).expand_dims(trace=["one", "two"])
    result = StepExponentialAnalysis.run(data, coords="x")
    np.testing.assert_allclose(result.params.c, 2)
    np.testing.assert_allclose(result.params.tau, 3, rtol=0.02)
    assert np.isfinite(result.params_std.tau).all()
    assert np.isnan(result.params_std.c).all()
    assert (result.debug_results.edge_lower < result.params.c).all()
    assert result.analysis_class.endswith("StepExponentialAnalysis")


def test_step_input_validation_and_failed_trace_isolation():
    x = np.linspace(0, 10, 101)
    y = np.where(x >= 2, 0.1 + np.exp(-np.maximum(x - 2, 0) / 3), 0.0)
    data = xr.concat([array(x, y), array(x, np.full_like(x, np.nan))], dim="trace")
    result = StepExponentialAnalysis.run(data, coords="x")
    assert result.success.values.tolist() == [True, False]
    with pytest.raises(ValueError):
        StepExponentialAnalysis.run(data, coords="x", guess={"tau": -1})
    with pytest.raises(ValueError):
        StepExponentialAnalysis.run(data, coords="x", bounds={"tau": (2, 1)})


def test_integral_regression_delta_covariance_matches_numerical_estimator_jacobian():
    x = np.linspace(0, 6, 31)
    y = 1.2 * np.exp(-x / 2) + 0.2 + np.random.default_rng(9).normal(0, 0.005, len(x))
    result = ExponentialRegressionAnalysis.run(array(x, y), dim="x")
    h, columns = 1e-5, []
    for i in range(len(x)):
        delta = np.zeros_like(y)
        delta[i] = h
        plus = ExponentialRegressionAnalysis.run(
            array(x, y + delta), dim="x"
        ).fit_params
        minus = ExponentialRegressionAnalysis.run(
            array(x, y - delta), dim="x"
        ).fit_params
        columns.append(
            [(float(plus[n]) - float(minus[n])) / (2 * h) for n in ["k", "a", "b"]]
        )
    jac = np.array(columns).T
    prediction = ExponentialRegressionAnalysis.func(
        array(x, y).x, **result.fit_params
    ).values
    sigma2 = np.sum((y - prediction) ** 2) / (len(x) - 3)
    np.testing.assert_allclose(
        result.debug_results.covariance, sigma2 * (jac @ jac.T), rtol=1e-4, atol=1e-12
    )
    assert float(result.params_std.decay_constant) > 0
