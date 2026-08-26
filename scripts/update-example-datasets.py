# %% [markdown]
# A script for updating / fixing the example datasets.

# %%
# ruff: noqa

# %load_ext autoreload
# %autoreload all

from pathlib import Path
import xarray as xr
import hvplot.xarray

from sqe_analysis.example_data import get_dataset_names, open_dataset


def save_updated(ds: xr.Dataset, ds_name: str):
    ds_path = Path("../src/sqe_analysis/example_data") / Path(
        ds_name + "-update"
    ).with_suffix(".nc")

    assert not ds_path.exists(), str(ds_path)

    print(f"Saving to {ds_path}")
    ds.to_netcdf(ds_path, engine="h5netcdf")


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
    ds_path = Path("../src/sqe_analysis/example_data") / Path(
        ds_name + "-update"
    ).with_suffix(".nc")

    assert not ds_path.exists(), str(ds_path)

    print(f"Saving to {ds_path}")
    ds.to_netcdf(ds_path, engine="h5netcdf")

# %% [markdown]
# ## Fix some quality notes for T1 data

# %%
ds_names = [n for n in get_dataset_names() if n.startswith("t1") and "QM" in n]
ds_names

# %%
from sqe_analysis.signal_processing import project_complex

# %%
for ds_name in ds_names:
    ds = open_dataset(ds_name).pipe(project_complex)
    print(ds_name)
    display(ds.hvplot(x="idle_time"))  # , logx=True))
    print(ds.quality_notes)
    print("\n\n\n")

# %%
[
    "t1-high_low_snr-RX4_QM_30",
    "t1-high_low_snr_cut_off-RX4_QM_29",
    #'t1-high_medium_snr_cut_off-RX4_QM_28',
    #'t1-high_snr-RX4_QM_34',
]

# %%
ds_name = "t1-high_low_snr-RX4_QM_30"
ds = open_dataset(ds_name)

display(ds.pipe(project_complex).hvplot(x="idle_time"))

(
    ds.assign_attrs(
        # quality_notes="Good SNR on Q41, very low SNR on Q40, Q42 and Q43. The maximum idle time is a little bit short."
        quality_notes="Good SNR on Q41, No signal on Q40, Q42 and Q43. The maximum idle time is a little bit short."
    )
    # .pipe(save_updated, ds_name)
)

# %%
ds_name = "t1-high_low_snr_and_no_signal_cut_off-RX4_QM_29"
ds = open_dataset(ds_name)

display(ds.pipe(project_complex).hvplot(x="idle_time"))

(
    ds.assign_attrs(
        quality_notes="Good SNR on Q41 and Q43, no signal on Q40 and Q42. There is a glitch at short idle durations in the first five data points. The maximum idle time is a bit short, decay doesn't fully saturate."
    )
    # .pipe(save_updated, ds_name)
)

# fix typo
del ds.attrs["qualiy_notes"]

ds.pipe(save_updated, ds_name)
