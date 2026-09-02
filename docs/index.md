---
orphan: true
tocdepth: 2
file_format: mystnb
kernelspec:
  name: python3
---

# sqe-analysis

sqe-analysis is a small, well-crafted, and practical data analysis toolbox for your superconducting quantum circuit experiments.
Welcome to the sqe-analysis documentation!

These are the main principles of sqe-analysis:
- **If the SNR is good, the analysis should *just work*.** Ideally, the analysis should require no input parameters apart from the data.
- **Simple, consistent API.** [Keeping things simple](https://en.wikipedia.org/wiki/KISS_principle) lowers the barrier for using the library, and makes it easier to understand for both busy researchers and LLMs. Additionally, simplicity keeps maintenance burden low. This is important for a library to survive in an academic environment with not much resources for maintenance, and where maintainers change often.
- **Core functionality built on top of [Xarray](https://xarray.dev/).** Xarray is a simple but extremely powerful layer on top of Numpy arrays which makes it easy to deal with experimental data.
- **Example data.** The library includes real experimental data which is used in unit tests. Importantly, the examples also include data with no signal, so that we can test that failures are reported correctly. If an experiment produces data that looks correct, but the analysis fails, the data set should be added to the examples and the analysis should be fixed so that it works with both the new and the existing data.
- **No visualization.** The sqe-analysis library deliberately does not come with any built-in visualization. Different experimental frameworks use different plotting libraries, and supporting all of them would add significant complexity. Additionally, there is no single obvious way to visualize a given type of data analysis. The appropriate visualization will depend on the data shape, and analysis may consist of multiple internal analysis steps, and having visualizations for those internal steps is pointless. Visualization is therefore the responsibility of the experimental framework that uses sqe-analysis.
- **Minimal dependencies.** Similarly to simplicity, having as few dependencies as possible makes the code easier to understand and to maintain long term, as there is less need for updates. Currently, the only required dependencies of the library are Scipy and Xarray.
- **Functional programming style.** The analysis has no internal mutable state. This makes testing and debugging easier.

To give you a flavor of sqe-analysis, here is an example T1 fit, using one of the built-in example datasets.

```{code-cell} python
:tags: [remove-cell]
# set some display options, not shown in rendered page
import xarray as xr
xr.set_options(display_expand_data=False)
```

```{code-cell} python
from sqe_analysis.example_data import open_dataset as open_example_dataset
from sqe_analysis.analysis import ExponentialRegressionAnalysis
from sqe_analysis.signal_processing import project_complex
```

Load an example dataset:
```{code-cell} python
# project_complex projects complex-valued data to the real axis
data = project_complex(
    open_example_dataset("t1-high_snr-RX4_QM_34").Q40
)
data
```

Run the analysis:
```{code-cell} python
result = ExponentialRegressionAnalysis.run(data)
result
```

Visualize the data and the evaluated fit.
Visualization should be handled by an external library.
In this case, we use [hvPlot](https://hvplot.holoviz.org), which is not a required dependency of sqe-analysis.
```{code-cell} python
import hvplot.xarray

fit_eval = ExponentialRegressionAnalysis.func(
    data.idle_time,
    **result.fit_params,
)

(
  data.hvplot.scatter(x="idle_time", label="Data")
  *
  fit_eval.drop_attrs().rename("fit result").hvplot(x="idle_time", label="Fit")
)
```

To get started, see the [user guide](guide/index), or if you want to look up the documentation for a specific function, see the [API reference](api/index).

sqe-analysis is free and open source software available under the EUPL license. See [the repository](https://github.com/RIKEN-SQERT/sqe-analysis/) for more information. sqe-analysis was initially developed at the [Superconducting Quantum Electronics Research Team](https://www.riken.jp/en/research/labs/rqc/superconduct_qtm_electron/index.html) at RIKEN.

```{toctree}
:maxdepth: 2
:hidden:

guide/index
api/index
```
