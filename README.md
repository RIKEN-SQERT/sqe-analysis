# sqe-analysis - Data analysis library for superconducting quantum circuit experiments

sqe-analysis is a small, well-crafted, and practical data analysis toolbox for your superconducting quantum circuit experiments.

For additional information, see [the documentation](https://sqe-analysis.readthedocs.io/).

Reflection and notch resonator fits are available as
`ReflectionResonatorAnalysis` and `NotchResonatorAnalysis` from
`sqe_analysis.analysis`. They accept complex Xarray DataArrays and return the
same `CurvefitAnalysisResult` used by the other curve-fitting analyses, with
Q values, resonance frequency, linewidth and statistical uncertainties.
See the [resonator guide](docs/guide/resonators.md) for examples and conventions.

Real peak, double-peak, oscillation, beating, Gaussian-windowed fringe, linear,
and stepped-exponential analyses use the same Xarray/result API. Ranked peak/dip
detection lives in `sqe_analysis.signal_processing.detect_peaks`. Curve fits now
return marginal statistical errors in `params_std` and full covariance in
`debug_results.covariance`. See the [model and uncertainty guide](docs/guide/standard-models.md)
for the API mapping, executable examples, units, and unavailable-error cases.


## License

The sqe-analysis library is Copyright 2026 the contributors to sqe-analysis, licensed under the EUPL v1.2, see [LICENSE](./LICENSE).
