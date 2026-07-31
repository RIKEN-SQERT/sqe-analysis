"""
Example datasets bundled with the library

This module contains example data that can (and should) be used in documentation
and unit tests. The data should come from real experiments, so that the fitting
algorithms can be evaluated and tested against a ??realistic scenario??.

The data should be in NetCDF 4 format. The axes should be labeled and have units.
Each dataset should have the following NetCDF metadata:

- title: string
    - a short, one-sentence human-readable description of the data. For example:
        - "Resonator spectroscopy"
        - "Resonator spectroscopy vs drive power"
- description: string
    - a more detailed description of the data, in a few sentences. An
    experienced researcher should be able to imagine what the data looks like
    based on this description. Examples:
        - "Complex-valued S21 transmission of a superconducting resonator capacitively coupled to a transmon qubit. Measured with a VNA."
        - "Complex-valued readout resonator transmission when a qubit is prepared in the excited state as a function of idle time for T1 measurement"
- quality_notes: string
    - Human-readable description of the expected fit result and data
    quality, such as "Clean resonance at 6.123 GHz", "Should contain four
    resonances but only contains three", "Very faint signal at 4.321 GHz", or
    "No signal at all, just noise"
- source: string
    - A description of where the data came from. For example:
        - a DOI link to a zenodo dataset: "https://doi.org/10.5281/zenodo.12345678"
        - an unambiguous reference to a measurement dataset: "RIKEN, SQERT, fridge X, cooldown N, chip ID Y, dataset ID Z"
- source_type: string
    - "experimental" or "simulation"
- expected_fit_result: JSON dictionary (optional)
    - A string encoding a JSON dictionary of machine-readable expected fit
    values that can be used in unit tests. The values should be in SI base
    units. For example,
    `{"resonance_frequency": 6.123e9, "linewidth": 5e6}`
- license: string
    - A license identifier such as "CC BY-SA 4.0"

The fields in the metadata are required unless marked as optional.

The file names should match the following pattern:
`[title]-[qualifier]-[label].nc`, with underscores instead of spaces. The title
should be similar to the title in the metadata (with underscores). The qualifier
should be similar to the "notes" metadata field, such as 'clean_resonance' or
'low_snr'. The label should be an abbreviated version of the "source" metadata
field, in order to distinguish multiple datasets with the same title and
qualifier. Hyphens should not be used apart from separating the three parts. For
example:
- `resonator_spectroscopy-good_snr-RIKEN_XLD4_XLD4_64QFY2023_10_20260729_008.nc`
- `resonator_spectroscopy_vs_power-no_signal-riken_XLD4_64QFY2023_10_20260729_009.nc`
"""
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
# - helper function to verify that the metadata matches the schema
