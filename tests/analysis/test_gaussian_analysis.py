"""
Tests for GaussianAnalysis
"""

from sqe_analysis.analysis import GaussianAnalysis
from xarray.testing import assert_allclose
from sqe_analysis.signal_processing import project_complex
from util import open_test_dataset

def test_gaussian_analysis_high_snr():
    data, _, _ = open_test_dataset("ac_stark_shift_vs_resonator_frequency-high_snr-RX3_1757")
    data = data.Q72

    # shorthands for dimension names
    f_q = "qubit_drive_frequency_shift"
    f_r = "resonator_drive_frequency_shift"

    # use minimum location as approximate expected fit result
    expected = (
        data.pipe(project_complex, dim=f_q)
        .pipe(lambda y: y - y.median([f_q, f_r]))
        .pipe(abs)
        .idxmax(f_q)
    )

    # TODO: remove 1e9 normalization, shouldn't be required (after we implement guess)
    result = GaussianAnalysis.run(data / 1e9, coords=f_q)
    c = result.params.c

    #assert_allclose(c, expected, rtol=0.5, atol=1)
    assert_allclose(
        c.isel({f_r: slice(None, 10)}),
        expected.isel({f_r: slice(None, 10)}),
        # the step size is around 3MHz so this is an acceptable tolerance
        atol=2.8,
    )
