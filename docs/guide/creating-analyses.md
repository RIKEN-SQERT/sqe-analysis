---
file_format: mystnb
kernelspec:
  name: python3
---

# Creating a custom analysis class

This page explains how to create a custom analysis class.
If you think others might find your analysis class useful, please consider making a pull request [on GitHub](https://github.com/RIKEN-SQERT/sqe-analysis/) incorporating your addition!
<!-- TODO: link to contributor guide (including contribution to example data) -->

Note that curve fitting is a kind of data analysis, but not all data analysis is curve fitting.
We will look at general data analysis first, and then discuss curve fitting separately.


## A simple example

In this toy example, we will create an analysis class which picks the maximum location of a dataset along a given dimension.

First, let's create some example data that we will analyze.
We will be using [holoviews](https://holoviews.org/) and [hvPlot](https://hvplot.holoviz.org/) for visualization in these examples, but note that sqe-analysis is independent of any particular visualization library, and all analysis classes work without having hvPlot installed.
```{code-cell} python
import xarray as xr
import numpy as np
import holoviews as hv
import hvplot.xarray

x = np.linspace(0, 10, 51)
peak_data = xr.DataArray(
      np.exp(-(x - 4) ** 2 / 0.5),
      coords=[("x", x)],
      # the data array must have a name
      # for plotting to work with hvplot
      name="peak data",
)

peak_data.hvplot(x="x")
```

At the most basic level, an analysis class should inherit {py:class}`~sqe_analysis.analysis_base.BaseAnalysis` and implement the {py:meth}`~sqe_analysis.analysis_base.BaseAnalysis.run` method.
The method should return an {py:class}`~sqe_analysis.result.AnalysisResult` object, which has a few required attributes.
```{code-cell} python
from sqe_analysis.analysis_base import BaseAnalysis
from sqe_analysis.result import AnalysisResult, get_source_dataset_id

class SimplePeakLocationAnalysis(BaseAnalysis):
    """
    A toy example analysis which returns the location of the maximum of the data.
    """
    @classmethod
    def run(cls, data: xr.DataArray, dim: str) -> AnalysisResult:
        # perform the analysis
        peak_location = data.idxmax(dim)

        # collect the result to an AnalysisResult object
        return AnalysisResult(
            # Params is a Dataset
            # (https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html),
            # which has one key for each parameter, or quantity of interest.
            params=xr.Dataset({
                "peak_location": peak_location,
            }),
            
            # Indicate whether the analysis was a success. In this example, it
            # always is. By passing coords=..., we ensure that the success
            # indicator will have the same shape as the result, even for
            # multi-dimensional data.
            success=xr.DataArray(True, coords=peak_location.coords),

            # We have to tell the result object which class created it, for
            # reproducibility
            analysis_class=cls,

            # The result object should also know which dataset it came from.
            # Here, we have not defined a 'dataset_id' for our example data,
            # so we will get a warning from the get_source_dataset_id helper
            # function. For real experimental data, the experiment framework
            # should attach a unique ID to the data for reproducibility.
            source_dataset_id=get_source_dataset_id(data),
        )
```

Note that the `run` method is a [class method](https://docs.python.org/3.11/library/functions.html#classmethod).
This means that it is called like `SimplePeakLocationAnalysis.run(...)` instead of `SimplePeakLocationAnalysis().run(...)`.
This way, the analysis function cannot use intermediate variables like `self.something = ...`.
While this may seem limiting at first, this is advantageous because it makes testing and debugging easier. <!-- TODO: link to page discussing functional programming style -->

We can now test our analysis:
```{code-cell} python
peak_result = SimplePeakLocationAnalysis.run(peak_data, dim="x")
peak_result
```
The result object has a nice HTML representation which allows easily checking the values.
Note the warning about the missing dataset ID, as expected.


The quantities of interest can be accessed via `.params`:
```{code-cell} python
peak_result.params
```

```{code-cell} python
peak_result.params.peak_location
```

Let's visualize the result.
```{code-cell} python
(
    peak_data.hvplot(x="x")
    *
    hv.VLine(x=peak_result.params.peak_location.item()).opts(color="k")
)
```
That is essentially it!
See the {py:class}`~sqe_analysis.result.AnalysisResult` class documentation for additional information that can be included in the analysis result.



Note that in a proper analysis class, the `dim` argument should default to `None` and use the {py:func}`~sqe_analysis.xarray_util.longest_dim` function to choose the longest dimension by default.


### Multidimensional analysis

Since we're using Xarray, the analysis will automatically work for multidimensional data, as long as we use only Xarray-compatible functions in the `run()` method, such as [`idxmax`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.idxmax.html) used above.

```{code-cell} python
peak_locs  = np.array([3, 5, 7])
peak_data_multi = xr.DataArray(
    np.exp(-(x[:, np.newaxis] - peak_locs) ** 2 / 0.5),
    coords=[("x", x), ("c", peak_locs)],
    name="multi-peak data",
)

peak_data_multi.hvplot(x="x", by="c")
```

```{code-cell} python
peak_result_multi = SimplePeakLocationAnalysis.run(peak_data_multi, dim="x")
peak_result_multi
```
Note how the call to `run()` is identical to the previous case, but the resulting `peak_location` is now an array with `c` as a coordinate.


```{code-cell} python
(
    peak_data_multi.hvplot(x="x", by="c")
    *
    hv.Overlay([
        hv.VLine(x.item())
        for x in peak_result_multi.params.peak_location
    ])
)
```


## Curve fitting

In this example, we will implement an analysis class which performs curve fitting.

sqe-fitting comes with a specialized subclass of `BaseAnalysis`, {py:class}`~sqe_analysis.analysis_base.CurvefitAnalysis`, which is suited for analysis that consists of a single curve fit.
This class has a pre-defined {py:meth}`~sqe_analysis.analysis_base.CurvefitAnalysis.run` method, which internally calls the [Xarray curvefit](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.curvefit.html) function.

The minimum requirement for a `CurvefitAnalysis` is simply defining the model function that we want to fit, by overriding the {py:meth}`~sqe_analysis.analysis_base.CurvefitAnalysis.func` method:
```{code-cell} python
from sqe_analysis.analysis_base import CurvefitAnalysis

class LineFit(CurvefitAnalysis):
    @classmethod
    def func(cls, x, a, b):
        return a * x + b
```
That's it! We can now use the `run()` method to perform curve fitting:
```{code-cell} python
x = np.linspace(-5, 5, 21)
line_data = xr.DataArray(
    # add some noise to make the fitting a bit more interesting
    2 * x + 1 + np.random.default_rng(seed=42).normal(size=x.size, scale=0.5),
    coords=[("x", x)],
    name="line data",
)

line_result = LineFit.run(line_data, coords="x")
line_result
```

In addition to the `params` as above, the result has a `fit_params` attribute, which can be used to evaluate the model function:
```{code-cell} python
# evaluate the fit at a sparser set of points
x_eval = xr.DataArray(coords=[("x", np.linspace(-5, 5, 11))]).x

fit_eval = LineFit.func(
    x_eval,
    **line_result.fit_params,
)

(
    line_data.hvplot.scatter(x="x", label="data")
    # have to add a name for hvplot to work
    * fit_eval.rename("fit").hvplot(x="x", label="fit")
)
```

In this example, `params` and `fit_params` are the same, but we will see below that they can be different, if additional derived quantities are added to `params`.

### Adding derived quantities to the fit result

**TODO**
```
class LineFitWithIntercept(LineFit):
    ...
```

### Adding an initial guess

**TODO**
```
class LineFitWithGuess(LineFit):
    ...
```

### Curve-fitting two-dimensional data (surface fitting)

**TODO**

## Complex-valued data

**TODO**

## Multi-step analysis

In this example, we will create an analysis class that combines several analyses.
It first finds the peak locations along one dimension using the `SimplePeakLocationAnalysis` class defined above, and then fits a line to the found peak locations.
Note that this is *not* a subclass of `CurvefitAnalysis` (even though we are performing curve fitting), since it involves multiple steps.
`CurvefitAnalysis` is intended only for cases where the entire analysis consists of fitting a single curve.


**TODO**
