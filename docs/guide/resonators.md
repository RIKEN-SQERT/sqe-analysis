---
file_format: mystnb
kernelspec:
  name: python3
---

# Reflection and notch resonators

`ReflectionResonatorAnalysis` and `NotchResonatorAnalysis` use the same
Xarray input and result objects as the other analyses. They fit the **complex,
linear-amplitude** response jointly in its real and imaginary components.
They do not require a database, instrument driver or measurement framework.

Choose the model from the physical circuit topology:

- **Reflection:** a one-port reflection response. A circulator can route this
  response to a VNA transmission port, so a trace called S21 can still require
  the reflection model.
- **Notch:** a hanger/notch resonance in a transmission feedline.

Magnitude in dB or phase alone cannot replace the complex input. If your data
consists of linear I and Q arrays, form `I + 1j * Q` before calling the analysis.
Convert dB magnitude and phase to complex amplitude explicitly if needed.

## A single trace

```{code-cell} python
import numpy as np
import xarray as xr
from sqe_analysis.analysis import (
    ReflectionResonatorAnalysis,
    NotchResonatorAnalysis,
)

frequency = np.linspace(7.998e9, 8.002e9, 301)
response = ReflectionResonatorAnalysis.func(
    frequency,
    fr=8.0001e9,
    Ql=40000,
    Qc=70000,
    amplitude=1.3,
    phase=0.4,
    delay=65e-9,
    phi=0.1,
    reference_frequency=8e9,
)
rng = np.random.default_rng(42)
response += 0.0002 * (
    rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
)
data = xr.DataArray(
    response,
    dims="frequency",
    coords={"frequency": frequency},
    name="reflection",
    attrs={"dataset_id": "synthetic-reflection-example"},
)
data.frequency.attrs["units"] = "Hz"

result = ReflectionResonatorAnalysis.run(data, coords="frequency")
result.params[["fr", "Qc", "Qi", "Ql", "linewidth"]]
```

The result is a `CurvefitAnalysisResult`. Its fields follow the existing API:

- `params`: fitted and derived quantities.
- `params_std`: statistical standard errors, including propagated errors of
  internal Q and linewidth, using the full covariance matrix.
- `success`: a boolean for each trace, including numerical and quality checks.
- `fit_params`: arguments for evaluating the model. Use this field, rather
  than `params`, because `params` also includes derived quantities.
- `fit_params_guess`: the actual initial guess of the winning multistart fit.
- `intermediate_results.fitted_data`: fitted response on the input grid.
- `intermediate_results.residual`: complex measured-minus-fitted response.
- `intermediate_results.relative_residual`: complex residual RMS divided by
  the fitted resonance-circle diameter, rather than the background amplitude.
- `debug_results.fit_covariance`: covariance in the physical fit parameters,
  with labeled `_cov_i` and `_cov_j` dimensions.
- `debug_results.delta_bic`: BIC improvement over a gain/delay-only background.

```{code-cell} python
result.success
```

```{code-cell} python
result.params_std[["fr", "Qc", "Qi", "linewidth"]]
```

Evaluate the model as with other curve-fitting analyses:

```{code-cell} python
fitted = ReflectionResonatorAnalysis.func(data.frequency, **result.fit_params)
fitted
```

This also works on a new, denser Xarray frequency coordinate. No plotting
library is required; the result can be used with Matplotlib, hvPlot or other
visualization libraries.

## Power sweeps and other additional dimensions

All dimensions except the selected frequency dimension are preserved. They
can represent power, time, devices, states, or any combination of these.

```{code-cell} python
sweep = xr.concat(
    [data, 1.1 * data],
    dim=xr.IndexVariable("power", [-90, -80]),
)
sweep_result = ReflectionResonatorAnalysis.run(sweep, coords="frequency")
sweep_result.params.fr
```

These synthetic traces only differ in gain; their frequencies should agree.
Real power sweeps can have power-dependent frequency and loss. Check each
trace's success flag and inspect its shape before interpreting those changes.

`coords=None` selects the longest dimension. Explicit `coords="frequency"`
is recommended whenever a sweep dimension might be longer than the frequency
axis. Frequency must be a strictly increasing, finite, positive 1-D dimension
coordinate in **Hz**. Nonuniform spacing is supported. Compute Dask-backed
data before fitting; lazy/distributed execution is not implemented.

## Initial guesses, bounds and failures

As in other analyses, manual values override automatic guesses and default
bounds. Values can be scalars or DataArrays over the non-frequency dimensions.

```{code-cell} python
constrained = ReflectionResonatorAnalysis.run(
    sweep,
    coords="frequency",
    guess={"fr": 8.0001e9, "Ql": 40000, "Qc": 70000},
    bounds={"fr": (8.0e9, 8.0002e9)},
    curvefit_kwargs={
        "errors": "raise",
        "kwargs": {"max_nfev": 4000},
    },
)
constrained.success
```

The backend is SciPy `least_squares`, rather than the real-valued Xarray
`curvefit` backend. The shared `curvefit_kwargs` argument supports:

- `errors="ignore"` (default) or `"raise"` for individual fit failures;
- `skipna=True` (default) to omit samples with non-finite real or imaginary parts;
- `kwargs` with `max_nfev`, `ftol`, `xtol`, `gtol`, and `verbose` optimizer options.

Other Xarray curvefit options are rejected explicitly. Positive Q/amplitude
domains and mismatch-phase bounds strictly within `(-pi/2, pi/2)` are required.
Frequency bounds must overlap the data. Malformed input or configuration
raises even with `errors="ignore"`. With that setting, an individual failed
trace has NaN parameters and `success=False`, without aborting valid neighbors.
An optimized trace failing quality checks retains its parameters for inspection.

## Models, conventions and limits

Both classes use

```text
S(f) = A * exp(i * [phase - 2*pi*(f - f_ref)*delay])
       * [1 - c*(Ql/Qc)*(1 + i*tan(phi)) / (1 + 2i*Ql*(f - fr)/fr)]
```

Here `c=2` for reflection and `c=1` for notch transmission. `phase` and `phi`
are in radians; delay is in seconds. `reference_frequency` is fixed to the
midpoint of each trace's valid frequency range and returned in `fit_params`.
The phase convention therefore avoids fitting an enormous absolute-frequency
phase offset. Its standard error is zero because the reference is fixed.

`Qc` is the **real, diameter-corrected external Q**, with
`1/Qi = 1/Ql - 1/Qc`. The inverse complex coupling is
`(1 + i*tan(phi))/Qc`; consequently the magnitude of the complex coupling Q
is `Qc*cos(phi)`. Compare like conventions when comparing to another package.
The loaded full width at half maximum is `linewidth = fr/Ql` in Hz.
The complex-coupling/diameter correction convention is discussed by
[Khalil et al.](https://arxiv.org/abs/1108.3117) and
[Probst et al.](https://arxiv.org/abs/1410.3365).

Automatic fitting tries multiple coupling ratios and frequency initializations.
Successful optimization alone does not imply a usable resonator result:

- Q values must be physical (`Qc > Ql > 0`), and covariance identifiable.
- Relative standard errors of Ql and Qc must be at most 50% by default
  (`max_relative_std=0.5`); tighten this for precision work.
- The solution must be interior to its bounds and frequency window.
- The linewidth must cover at least two median sample intervals by default
  (`min_linewidth_points=2`). Use more samples for precision measurements.
- Residual RMS must be no more than 10% of the fitted resonance diameter by
  default (`max_relative_residual=0.1`). This avoids hiding mismatch of weak
  resonances behind a large background signal.
- The resonator model must improve BIC over a gain/linear-phase background
  by more than 10, to reject flat or delay-only responses.

Covariance errors assume independent, equal-variance Gaussian errors in the
real and imaginary components. They do not include systematic uncertainty.
Always inspect measured/fitted magnitude, phase, and complex response, and
check repeatability and window dependence. An isolated, linear, sufficiently
sampled resonance is assumed. Saturation, hysteresis, drifting resonance,
multiple resonances and arbitrary additive leakage are not modeled.
No data acquisition, YAML update, QCoDeS database access or hardware side
effect occurs in these analyses.
