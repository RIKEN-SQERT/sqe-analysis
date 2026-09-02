# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Example data import, cleanup & annotation
#
# This notebook shows an example of importing example data from an external experiment framework and annotating it with the appropriate metadata. It can be used as a template for performing data import and cleanup. This python script can be converted to a Jupyter notebook using Jupytext.
#
# Doing this kind of annotation is tedious and difficult to automate. This is the kind of task that is well suited for LLMs. Here is an example prompt for a coding agent to use this script as a template. Place your example data in the `exported_data/` folder and change the file name in the prompt. Please always carefully check the output.
#
# <!-- TODO: this should probably be an agent skill instead... -->
# ```
# Take a look at the @scripts/example-data-import.py script. Create a copy of it in the folder @scripts/exported_data/ and modify it to instead use the data set @scripts/exported_data/**[YOUR-DATASET]**. Check the metadata, and create a plot of the data using matplotlib and look at it, and update the description and metadata of the example dataset accordingly. See @src/sqe_analysis/example_data/__init__.py for information on the metadata format. Look at the descriptions and quality notes of the existing example datasets in @src/sqe_analysis/example_data/. Propose a file name for the newly created example data set but don't save it yet, I will check the output before saving it.
#
# Use `uv run <modified script>` inside the scripts folder to run your script. Remember to update the relative file paths in the modified script.
# ```

# +
from pathlib import Path

import hvplot.xarray
import xarray as xr

from sqe_analysis.example_data import open_dataset as open_example_dataset
from sqe_analysis.example_data import validate_metadata

LICENSE = "CC BY-SA 4.0"
# -

# data has been exported from the measurement framework to exported_data/ folder
ds = xr.open_dataset("exported_data/1373-CheckKappaChi_Q72.nc")
ds

# check what the data looks like, so we can refer to it while writing the description
ds[next(iter(ds.data_vars))].real.hvplot(
    x="resonator_drive_frequency_shift",
    y="qubit_drive_frequency_shift",
).layout().cols(1)

# +
ds_annotated = (
    ds
    .drop_attrs(deep=False) # remove top-level attributes but keep e.g. units in coordinate axes
    .assign_attrs(
        title="AC stark shift vs resonator frequency and qubit state",
        description=(
            "Resonator transmission after populating the resonator and applying a qubit probe pulse, as a function of the resonator population pulse frequency and qubit probe pulse frequency, with the qubit initially prepared in g & e. "
            "Complex-valued three-dimensional data. "
            "Also known as chi-kappa-power measurement (see Sank et al. Phys. Rev. Applied 23, 024055 (2025)). "
            "Measured with QuEL-SE1. "
        ).strip(),
        quality_notes=(
            "Good SNR. There is some distortion around resonator drive frequency -30 MHz and +10 MHz with qubit in the e-state, and some bands along the qubit frequency shift near -10 to 0 MHz resonator frequency. "
        ).strip(),

        # based on the original metadata in the netcdf file
        source="RIKEN, SQERT XLD3 CD25 FY2023 144Q 3rd lot No6 dataset 1373",
        #author="Adrian Hesse / RIKEN",
        license=LICENSE,
        
        source_type="experimental",
    )
)

validate_metadata(ds_annotated)

ds_annotated
# -

# check metadata of existing example data for comparison
open_example_dataset("t1_flip-distorted-RX4_59")

# save the data in the examples folder of the library
path = Path("../src/sqe_analysis/example_data/ac_stark_shift_vs_resonator_frequency-good_snr_glitch-RX3_1373.nc")
assert not path.exists()
ds_annotated.to_netcdf(path, engine="h5netcdf")
