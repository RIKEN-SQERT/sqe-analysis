"""
Example datasets bundled with sqe-analysis

This module contains example data that can (and should) be used in documentation
and unit tests. The data should come from real experiments, so that the fitting
algorithms can be evaluated and tested against realistic conditions.

The data should be in NetCDF 4 format. The axes should be labeled and have
units. If a dataset has multiple traces (for example, form the simultaneous
measurement of multiple qubits), each trace should be a separate variable in the
dataset. Each dataset should have the following NetCDF metadata:

- ``title``: string
    - A short, one-sentence human-readable description of the data. For example:
        - "Resonator spectroscopy"
        - "Resonator spectroscopy vs drive power"
- ``description``: string
    - A more detailed description of the data, in a few sentences. An
      experienced researcher should be able to imagine what the data looks like
      based on this description. Examples:
      - "Complex-valued S21 transmission of a superconducting resonator capacitively coupled to a transmon qubit. Measured with a VNA."
      - "Complex-valued readout resonator transmission when a qubit is prepared in the excited state as a function of idle time for T1 measurement"
- ``quality_notes``: string
    - Human-readable description of the expected fit result and data
      quality, such as "Clean resonance at 6.123 GHz", "Should contain four
      resonances but only contains three", "Very faint signal at 4.321 GHz", or
      "No signal at all, just noise"
- ``source``: string
    - A description of where the data came from. For example:
        - a DOI link to a zenodo dataset: "https://doi.org/10.5281/zenodo.12345678"
        - an unambiguous reference to a measurement dataset: "RIKEN, SQERT, fridge X, cooldown N, chip ID Y, dataset ID Z"
- ``source_type``: string
    - "experimental" or "simulation"
- ``expected_fit_result``: JSON dictionary (optional)
    - A string encoding a JSON dictionary of machine-readable expected fit
      values that can be used in unit tests. The values should be in SI base
      units. For example, ``{"resonance_frequency": 6.123e9, "linewidth": 5e6}``.
      For datasets with multiple qubits, it should be a nested dictionary with
      top-level keys for the qubits:

        .. code-block::

            {
                "Q00": {
                    "resonance_frequency": ...,
                    "linewidth": ...
                },
                "Q01": {...},
                ...
            }

- ``license``: string
    - A license identifier such as "CC BY-SA 4.0"

The fields in the metadata are required unless marked as optional.

The file names should match the following pattern:
``[title]-[qualifier]-[label].nc``, with underscores instead of spaces. The title
should be similar to the title in the metadata (with underscores). The qualifier
should be similar to the "notes" metadata field, such as 'clean_resonance' or
'low_snr'. The label should be an abbreviated version of the "source" metadata
field, in order to distinguish multiple datasets with the same title and
qualifier. Hyphens should not be used apart from separating the three parts. For
example:

- ``resonator_spectroscopy-good_snr-RX4_008.nc``
- ``resonator_spectroscopy_vs_power-no_signal-RX4_009.nc``
"""

import json
from collections import defaultdict
from pathlib import Path

import xarray as xr

# TODO: 'the library' -> library name

# If the data becomes bigger than ~10 MB, it should be separated so it's not
# included with the wheel that comes with pip install, and is only included if
# you do pip install package[example-data]. There's two ways to do this:
# 1. create a separate -data package. For example, this is how it's done in
#    pangeo-forge / pangeo-data & dask / dask-data
# 2. download the data from the internet when it's requested. This is how
#    xarray.tutorial.open_dataset() does it.

# TODO: some ideas for utility functions
# - helper for browsing data (find by tag etc) (do we want tags?)


def get_dataset_names() -> list[str]:
    """
    Get a list of the names of the example datasets that can be passed to
    :py:func:`open_dataset`.
    """
    # All datasets are assumed to be NetCDF.
    return sorted(p.with_suffix("").name for p in Path(__file__).parent.glob("*.nc"))


def open_dataset(name: str) -> xr.Dataset:
    """
    Open an example dataset by name. The list of available names can be found
    with :py:func:`get_dataset_names`.
    """
    # All datasets are assumed to be NetCDF. Use h5netcdf for compatibility with
    # complex-valued data.
    ds_names = get_dataset_names()
    if name not in ds_names:
        raise ValueError(
            f"Name {name!r} not found in example dataset names {ds_names}."
        )

    try:
        return xr.open_dataset(
            (Path(__file__).parent / name).with_suffix(".nc"),
            engine="h5netcdf",
        )
    except ImportError as e:
        raise ImportError(
            f"Error while opening example data. Ensure that you have h5netcdf and h5py installed ({e})"
        ) from e


def validate_metadata(ds: xr.Dataset) -> None:
    """
    Verify that the metadata of an xarray Dataset matches the schema described in the documentation.

    Raises a ValueError if the schema doesn't match.
    """
    required_keys = [
        "title",
        "description",
        "quality_notes",
        "source",
        "source_type",
        "license",
    ]
    valid_keys = required_keys + ["expected_fit_result"]

    missing_keys = [k for k in required_keys if k not in ds.attrs]
    if missing_keys:
        raise ValueError(f"Required keys missing from metadata: {missing_keys}")

    invalid_keys = [k for k in ds.attrs if k not in valid_keys]
    if invalid_keys:
        raise ValueError(f"Invalid keys in metadata: {invalid_keys}")

    def fmt_dict(d: dict):
        return ", ".join([f"{k!r} of type {t}" for k, t in d.items()])

    # values of all required keys should be strings
    non_string_keys = {
        k: type(ds.attrs[k]) for k in required_keys if not isinstance(ds.attrs[k], str)
    }
    if non_string_keys:
        raise ValueError(
            f"Expected string values in metadata, but found {fmt_dict(non_string_keys)}"
        )

    empty_keys = [k for k in required_keys if not ds.attrs[k]]
    if empty_keys:
        raise ValueError(
            f"All required keys in the metadata should be non-empty, found empty values for {empty_keys}"
        )

    valid_source_types = ["experimental", "simulation"]
    source_type = ds.attrs["source_type"]
    if source_type not in valid_source_types:
        raise ValueError(
            f"'source_type' should be one of {valid_source_types}, but found {source_type!r}"
        )

    # check that the expected_fit_result has the expected format
    if "expected_fit_result" in ds.attrs:
        # NOTE: NetCDF only supports strings so the metadata should be a serialized JSON string
        expected_fit_result = json.loads(ds.attrs["expected_fit_result"])
        if not isinstance(expected_fit_result, dict):
            raise ValueError(
                f"Metadata 'expected_fit_result' should be a dict, found {type(expected_fit_result)}"
            )

        expected_fit_result_types = {k: type(v) for k, v in expected_fit_result.items()}
        all_dicts = all(v == dict for v in expected_fit_result_types.values())
        all_floats = all(v == float for v in expected_fit_result_types.values())
        if not (all_dicts or all_floats):
            raise ValueError(
                f"Expected 'expected_fit_result' metadata to be a dict of dicts or a dict of floats, but found {fmt_dict(expected_fit_result_types)}"
            )
        elif all_dicts:
            invalid_types = defaultdict(dict)
            for k, subdict in expected_fit_result.items():
                for k_sub, v_sub in subdict.items():
                    if not isinstance(v_sub, float):
                        invalid_types[k][k_sub] = type(v_sub)
            if invalid_types:
                raise ValueError(
                    f"If metadata 'expected_fit_result' is a dictionary of dictionaries, all values should be floats, but found {invalid_types}"
                )

            # the dict keys should match the data variables in the dataset
            if not sorted(expected_fit_result.keys()) == sorted(ds.keys()):
                raise ValueError(
                    f"Expected the 'expected_fit_result' metadata dict to have an entry for each data variable. Found data variables {list(ds.keys())} but metadata has {list(expected_fit_result.keys())}."
                )
