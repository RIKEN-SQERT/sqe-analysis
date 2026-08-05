"""
The main API of the library.
"""

from collections.abc import Hashable
from typing import cast

import xarray as xr

from sqe_fitting2.signal_processing import project_complex
from sqe_fitting2.xr_util import longest_dim


def analyze_time_of_flight(
    data: xr.Dataset,
    dim: Hashable | None = None,
    smoothing: int = 5,
) -> xr.Dataset:
    """
    ...

    Assumes that the data is demodulated.

    Always returns a result that is a value from the time axis of the dataset.

    Returns:
        A Dataset with the variable ???

    Example:

        >>> from sqe_fitting2.example_data import open_dataset
        >>> y = open_dataset("")

        Quick visualization of the result:
    """

    if dim is None:
        dim = longest_dim(data)

    # TODO: move to separate preprocessing step?
    proj = project_complex(data)
    smoothed = proj.rolling({dim: smoothing}, center=True).mean()
    diff = smoothed.differentiate(dim)

    step_locations = abs(diff).idxmax(dim)
    pre_step = proj.where(diff[dim] < step_locations)
    post_step = proj.where(diff[dim] > step_locations)
    signal = abs(post_step.median([dim]) - pre_step.median([dim]))
    noise = pre_step.std([dim])
    snr = signal / noise
    return cast(
        xr.Dataset,  # we know the output will be Dataset, because the input is Dataset
        xr.combine_by_coords(
            [
                step_locations.expand_dims(param=["step_location"]),
                snr.expand_dims(param=["SNR_estimate"]),
                # pre_step.expand_dims(param=["pre_step"]),
                # post_step.expand_dims(param=["post_step"]),
            ],
            join="outer",
        ),
    )
