# Getting started

## Installation

Currently, sqe-analysis is available via git. To install using [uv](https://docs.astral.sh/uv/) (recommended):
```shell
# activate your virtual environment
source your/virtual/environment/bin/activate
# install
uv pip install "sqe-analysis @ https://github.com/RIKEN-SQERT/sqe-analysis.git"
```

To load the example data, install the following additional dependencies:
```shell
uv pip install h5netcdf h5py
```

The Python version support of sqe-analysis follows the [Scientific Python Ecosystem support schedule](https://scientific-python.org/specs/spec-0000/).

## Running a simple analysis

**TODO** <!-- would this be just the same as what we already have on the index page? -->

To create additional analysis classes, see [the tutorial](./creating-analyses.md).
