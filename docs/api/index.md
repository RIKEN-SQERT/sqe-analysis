# API Reference

... here is the API reference ...

<!-- https://myst-parser.readthedocs.io/en/latest/syntax/code_and_apis.html#syntax-apis-sphinx-autodoc -->
<!-- TODO: check autosummary args (toctree, include-members __init__ (?), etc) -->
```{eval-rst}
.. autosummary::
  
  sqe_analysis.analysis
  sqe_analysis.analysis_base
  sqe_analysis.result
  sqe_analysis.xarray_util
```

<!--
```
  sqe_analysis.result
  sqe_analysis.signal_processing
  sqe_analysis.xarray_util
  sqe_analysis.example_data
```
-->

<!-- TODO: we should add :hidden:, but then we want something like autosummary to show the module name ... -->
```{toctree}
:maxdepth: 2
:hidden:

analysis
analysis_base
result
xarray_util
```
