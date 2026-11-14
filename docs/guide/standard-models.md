---
file_format: mystnb
kernelspec:
  name: python3
---

# Peaks, fringes, oscillations, and step responses

The following analyses use `Class.run(data, coords=..., guess=..., bounds=...,
curvefit_kwargs=...)` and return `CurvefitAnalysisResult`. Inputs are Xarray
DataArrays; outer dimensions are fitted independently. No acquisition software,
database, lmfit, or instrument object is required.

| Earlier operation | Public sqe-analysis API |
| --- | --- |
| `fit_ramsey_fringe` | `RamseyFringeAnalysis` |
| `dualGaussianFit` | `DoubleGaussianAnalysis` |
| `dualLorentzianFit2` | `DoubleLorentzianAnalysis` |
| `lorentzianFit2` | `LorentzianAnalysis` |
| `sindampBeatFit` | `DampedBeatingOscillationAnalysis` |
| `detect_peaks` | `sqe_analysis.signal_processing.detect_peaks` |
| `expheavisideFit` | `StepExponentialAnalysis` |
| `linearFit` | `LinearAnalysis` |
| `sinFit2` | `OscillationAnalysis` |

There are no legacy aliases or stateful `self.popt`/`self.perr` fields. Use named
entries in `result.params`, `result.params_std`, and `result.fit_params`.

## A peak fit and its statistical uncertainty

```{code-cell} python
import numpy as np
import xarray as xr
from sqe_analysis.analysis import LorentzianAnalysis

rng = np.random.default_rng(4)
x = np.linspace(-3, 3, 301)
y = 0.2 - 0.8 / (1 + ((x - 0.3) / 0.4)**2)
data = xr.DataArray(y + rng.normal(0, 0.005, len(x)), dims="frequency",
                    coords={"frequency": x}, attrs={"dataset_id": "example-peak"})
result = LorentzianAnalysis.run(data, coords="frequency")
print("Center:", result.params.c.item(), "+/-", result.params_std.c.item())
print("FWHM:", result.params.FWHM.item(), "+/-", result.params_std.FWHM.item())
prediction = LorentzianAnalysis.func(data.frequency, **result.fit_params)
assert result.success.item()
assert abs(result.params.c.item() - 0.3) < 0.02
```

The Lorentzian is `b + a / (1 + ((x-c)/sigma)**2)`. `a` is signed peak height,
not integrated area; `sigma` is half-width at half maximum and `FWHM=2*sigma`.
Gaussian component amplitudes also denote heights, but their sigma is a standard
deviation and `FWHM=2*sqrt(2*log(2))*sigma`.

`DoubleGaussianAnalysis` and `DoubleLorentzianAnalysis` use
`a1,c1,sigma1,a2,c2,sigma2,b`. A dominant single-peak fit seeds the residual search
for a second component. Guesses start with increasing centers, but component labels
can exchange during optimization; sort by center when assigning physical labels.
Overlapping/weak peaks can be unidentifiable even when optimization converges.
Supply manual guesses/bounds and inspect uncertainties and residuals.

```{code-cell} python
from sqe_analysis.analysis import DoubleGaussianAnalysis, LinearAnalysis

signal = 0.1 + np.exp(-0.5*((x+1)/0.25)**2) + 0.7*np.exp(-0.5*((x-1)/0.4)**2)
two_peaks = data.copy(data=signal + rng.normal(0, 0.003, len(x)))
double = DoubleGaussianAnalysis.run(two_peaks, coords="frequency")
print(double.params[["c1", "c2", "FWHM1", "FWHM2"]])
line = LinearAnalysis.run(data.copy(data=2*x+0.3+rng.normal(0, 0.01, len(x))),
                          coords="frequency")
print("Slope:", line.params.a.item(), "+/-", line.params_std.a.item())
```

Real-valued input retains its original baseline in the new real-trace models.
Complex input is projected to a real axis with `project_complex`; the resulting
signal is centered and may change sign. Returned parameters and errors refer to
that preprocessed signal. Preprocessed data, fitted data, and residuals are in
`intermediate_results`. Coordinates must be finite and strictly increasing.
For manual guesses/bounds, scalar values and outer-dimension DataArrays follow
the existing CurvefitAnalysis conventions. Automatic spectral guesses interpolate
irregular finite samples onto a uniform grid; manual seeds are advisable for
sparse/missing data or strongly nonuniform sampling.

## Ranked peak/dip detection

```{code-cell} python
from sqe_analysis.signal_processing import detect_peaks

peaks = detect_peaks(two_peaks, dim="frequency", dip=False, rank=3, prominence=0.1)
print(peaks[["position", "height", "strength", "index", "count"]])
assert peaks.position.sizes["peak"] == 3
```

`detect_peaks` is signal processing, not a curve fit: it returns an Xarray Dataset,
not an AnalysisResult. `dip=True` finds minima, `False` finds maxima, and `None`
chooses polarity from the largest median deviation per trace. `rank` sets the
fixed output size, allowing different peak counts in different outer traces.
Results are ordered by strength. Missing entries have NaN positions/heights and
index -1. `count` counts all accepted candidates; `peak_mask` marks them on the
original grid. Endpoints and candidates adjacent to missing samples are excluded.
No sub-sample position or position standard deviation is invented.

## Oscillations and beating

```{code-cell} python
from sqe_analysis.analysis import OscillationAnalysis, DampedBeatingOscillationAnalysis

t = np.linspace(0, 12, 601)
oscillation = xr.DataArray(0.2 + 0.7*np.cos(2*np.pi*(0.8*t+0.1)), dims="time",
                           coords={"time": t}, attrs={"dataset_id": "example-oscillation"})
osc = OscillationAnalysis.run(oscillation, coords="time")
print(osc.params[["f", "phi", "period", "half_period", "quarter_period"]])

beating_signal = 0.2 + 0.7*np.exp(-t/7)*(np.cos(2*np.pi*0.8*t)+np.cos(2*np.pi*1.3*t))
beating = DampedBeatingOscillationAnalysis.run(
    oscillation.copy(data=beating_signal + rng.normal(0, 0.002, len(t))), coords="time")
print("Second frequency:", beating.params.second_frequency.item(),
      "+/-", beating.params_std.second_frequency.item())
```

`OscillationAnalysis` fits `b+a*cos(2*pi*(f*x+phi))`; `phi` is in **turns**, as in
`DampedOscillationAnalysis`. Periods are coordinate intervals, not phase-adjusted
pi-pulse calibration values. Do not infer a pulse setting without checking phase
and the measured observable.

`DampedBeatingOscillationAnalysis` fits two tones with shared amplitude, decay,
and coordinate shift: `b+a*exp(-x/tau)*(cos(2*pi*f*(x-c))+cos(2*pi*(f+s)*(x-c)))`.
The frequencies are `f` and `f+s`; `c` has coordinate units, not phase units.
`second_frequency` uncertainty includes the correlation between f and s. Automatic
seeds work best when the tones are spectrally resolvable; provide manual guesses
for unresolved beats. This model does not fit independent amplitudes/phases/decays.

## Gaussian-windowed Ramsey fringes

```{code-cell} python
from sqe_analysis.analysis import RamseyFringeAnalysis

fringe_signal = 0.1 + 0.7*(1-0.6*np.cos(2*np.pi*(1.7*(t-6)+0.1)))*np.exp(-0.5*((t-6)/2)**2)
fringe = RamseyFringeAnalysis.run(oscillation.copy(data=fringe_signal), coords="time")
print("Envelope center:", fringe.params.c.item())
print("Selected extremum:", fringe.params.extremum_location.item())
```

The model is `b+a*(1-contrast*cos(2*pi*(f*(x-c)+phi)))*exp(-0.5*((x-c)/sigma)**2)`.
It also accepts a swept drive-frequency axis. In that case f is the **inverse** of
that axis's units (fringe spacing), not a qubit frequency in Hz. `c` is the Gaussian
envelope center. `extremum_location` is the global signed extremum of the fitted
curve inside the observed range and can differ from c; its standard deviation is
NaN because switches between fringe extrema make a local covariance error misleading.
`FWHM` describes the envelope. Default contrast bounds are [0,1] and may be overridden.

## A step followed by decay

```{code-cell} python
from sqe_analysis.analysis import StepExponentialAnalysis

response = np.where(t >= 2, 0.1+0.8*np.exp(-np.maximum(t-2, 0)/3), 0.)
step = StepExponentialAnalysis.run(
    oscillation.copy(data=response + rng.normal(0, 0.002, len(t))), coords="time")
print("Edge bracket:", step.debug_results.edge_lower.item(), step.debug_results.edge_upper.item())
print("Decay:", step.params.tau.item(), "+/-", step.params_std.tau.item())
assert np.isnan(step.params_std.c)
```

The model is `H(x-c)*(a+b*exp(-(x-c)/tau))`. Subtract a known pre-edge baseline and
pass real data. Fitting c with ordinary derivatives of a hard step is invalid.
Instead, the strongest sample jumps seed a discrete edge profile (`edge_candidates=8`
by default); a/b/tau are fitted conditionally at each edge and the lowest SSE wins.
`guess['c']` adds nearby grid candidates; c is always on the finite sample grid.
`edge_lower` and `edge_upper` bracket the unresolvable inter-sample onset. The edge's
standard deviation is NaN, not zero. a/b/tau errors are conditional on the selected
edge and exclude edge/model ambiguity. At least four post-edge samples are required.

The standard guess/bounds dictionaries accept scalars or outer DataArrays. Step
curvefit_kwargs supports `errors`, `skipna`, and nested `kwargs` containing
`max_nfev`, `ftol`, `xtol`, or `gtol`. Missing traces can fail independently without
aborting valid neighboring traces. No smooth rise-time parameter is implied.

## Standard deviations and covariance

The shared CurvefitAnalysis now sets `params_std` from the square roots of the
covariance diagonal and retains the full matrix in `debug_results.covariance`
(`cov_i`, `cov_j`). This also applies to existing Gaussian and damped-oscillation
fits. Gaussian FWHM errors are propagated; subclasses can implement
`extra_params_std(fit_params, covariance)` for additional quantities. Unknown
derived errors are omitted rather than set to zero.

Covariance follows SciPy/Xarray settings, including `absolute_sigma` when supplied
through solver kwargs. Without absolute sigma it is scaled by residual variance.
NaN means unavailable; infinity can indicate an unidentifiable fit. These are local
statistical errors, not total systematic uncertainties, error in complex projection,
or guarantees of model validity. Generic success flags describe optimizer output,
not a residual/uncertainty acceptance policy.

ExponentialRegressionAnalysis retains its integral estimator and point values.
For real data, it propagates independent equal-variance sample errors through that
estimator's analytic Jacobian (delta method), estimating noise from residuals with
n-3 degrees of freedom. This is not a nonlinear least-squares covariance. Complex,
missing-data, insufficient-sample, and degenerate cases have unavailable errors.
Existing resonator covariance behavior is retained.
