# %% [markdown]
# A script for updating / fixing the example datasets.

# %%
# %load_ext autoreload
# %autoreload all

from pathlib import Path
import xarray as xr
import hvplot.xarray

from sqe_fitting2.example_data import get_dataset_names, open_dataset

# %%
get_dataset_names()

# %% [markdown]
# ## Add units to `t1_flip` time data

# %%
ds_names = [n for n in get_dataset_names() if n.startswith("t1_flip")]

for ds_name in ds_names:
    ds = open_dataset(ds_name)
    if "units" not in ds.time.attrs:
        ds = ds.assign_coords(time=ds.time.assign_attrs(units="ns"))
    else:
        print(f"skipping {ds_name}, already has units {ds.time.units}")

    # add suffix, can't overwrite files in .to_netcdf()
    ds_path = Path("../src/sqe_fitting2/example_data") / Path(ds_name + "-update").with_suffix(".nc")
    
    assert not ds_path.exists(), str(ds_path)

    print(f"Saving to {ds_path}")
    ds.to_netcdf(ds_path, engine="h5netcdf")

# %%
