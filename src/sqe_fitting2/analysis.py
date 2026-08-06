"""
The main API of the library.
"""

from collections.abc import Hashable

import xarray as xr

from sqe_fitting2.analysis_base import BaseAnalysis
from sqe_fitting2.signal_processing import project_complex
from sqe_fitting2.xr_util import longest_dim


class TimeOfFlightAnalysis(BaseAnalysis):
    """
    Analysis for time-of-flight measurement

    Assumes that the data is demodulated.

    Always returns a result that is a value from the time axis of the dataset.

    Returns:
        A Dataset with the variable ???

    Example:

        >>> from sqe_fitting2.example_data import open_dataset
        >>> y = open_dataset("")

        Quick visualization of the result:
    """

    @classmethod
    def run(
        cls,
        data: xr.DataArray,
        dim: Hashable | None = None,
        smoothing: int = 5,
    ) -> xr.Dataset:
        # TODO: how should I split project_complex vs smoothing in preprocess??
        if dim is None:
            dim = longest_dim(data)

        proj = project_complex(data, dim=dim)
        smoothed = proj.rolling({dim: smoothing}, center=True).mean()
        diff = smoothed.differentiate(dim)

        # TODO: proper peak finding...
        step_locations = abs(diff).idxmax(dim)
        pre_step = proj.where(diff[dim] < step_locations)
        post_step = proj.where(diff[dim] > step_locations)
        signal = abs(post_step.median([dim]) - pre_step.median([dim]))
        noise = pre_step.std([dim])
        snr = signal / noise
        return xr.Dataset(
            dict(  # noqa: C408
                step_location=step_locations,
                snr=snr,
            )
        )
