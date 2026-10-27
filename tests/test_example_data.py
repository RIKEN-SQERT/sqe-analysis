import json

import pytest
import xarray as xr

from sqe_analysis.example_data import (
    get_dataset_names,
    open_dataset,
    validate_metadata,
)


def test_open_dataset_unknown_name():
    with pytest.raises(ValueError, match="not found"):
        open_dataset("nonexistent")


def test_all_metadata_valid():
    invalid = []
    for ds_name in get_dataset_names():
        try:
            validate_metadata(open_dataset(ds_name))
        except ValueError as e:
            invalid.append((ds_name, e))

    if invalid:
        raise ValueError("Errors validating metadata:\n" + "\n".join(f"{ds_name}: {e}" for ds_name, e in invalid))


def make_ds(var_names=("Q00",), expected_fit_result=None) -> xr.Dataset:
    """
    Build a dataset with minimal valid metadata, plus the given
    ``expected_fit_result``, serialized the way NetCDF stores it.
    """
    ds = xr.Dataset({name: ("x", [1.0, 2.0]) for name in var_names})
    ds.attrs = {
        "title": "T1",
        "description": "Qubit T1",
        "quality_notes": "Clean decay",
        "source": "RIKEN, RX4, 0",
        "source_type": "experimental",
        "license": "CC BY-SA 4.0",
        "author": "A. Author / Institute",
    }
    if expected_fit_result is not None:
        ds.attrs["expected_fit_result"] = json.dumps(expected_fit_result)
    return ds


def test_validate_metadata_without_expected_fit_result():
    validate_metadata(make_ds())


def test_validate_metadata_dict_of_floats():
    validate_metadata(make_ds(var_names=["Q00"], expected_fit_result={"t1": 1e-6}))


def test_validate_metadata_dict_of_lists_of_floats():
    validate_metadata(
        make_ds(var_names=["Q00"], expected_fit_result={"t1": [1e-6, 2e-6, 3e-6]})
    )


def test_validate_metadata_dict_of_mixed_floats_and_lists():
    validate_metadata(
        make_ds(
            var_names=["Q00"],
            expected_fit_result={"rising_edge": 1e-6, "falling_edge": [2e-6, 3e-6]},
        )
    )


def test_validate_metadata_dict_of_dicts():
    validate_metadata(
        make_ds(
            var_names=["Q00", "Q01"],
            expected_fit_result={"Q00": {"t1": 1e-6}, "Q01": {"t1": 2e-6}},
        )
    )


def test_validate_metadata_dict_of_dicts_with_lists():
    # Fit values from a 2D scan: one element per value of the sweep parameter
    validate_metadata(
        make_ds(
            var_names=["Q00", "Q01"],
            expected_fit_result={
                "Q00": {"t1": [1e-6, 2e-6], "t2_star": [3e-6, 4e-6]},
                "Q01": {"t1": [5e-6, 6e-6]},
            },
        )
    )


def test_validate_metadata_dict_of_dicts_with_nan():
    # Datasets where no signal is visible have NaN as the expected value
    validate_metadata(
        make_ds(var_names=["Q00"], expected_fit_result={"Q00": {"t1": float("nan")}})
    )


@pytest.mark.parametrize(
    ("var_names", "expected_fit_result"),
    [
        # not a dict
        (["Q00"], [1.0, 2.0]),
        (["Q00"], "t1"),
        # top level is neither a dict of dicts nor a dict of floats/lists
        (["Q00", "t2_star"], {"Q00": {"t1": 1e-6}, "t2_star": 2e-6}),
        (["Q00"], {"t1": "1e-6"}),
        (["Q00"], {"t1": None}),
        # values that are neither floats nor lists of floats
        (["t1"], {"t1": 1}),
        (["Q00"], {"Q00": {"t1": 1}}),
        (["Q00"], {"Q00": {"t1": "1e-6"}}),
        (["Q00"], {"Q00": {"t1": None}}),
        (["Q00"], {"Q00": {"t1": True}}),
        # empty lists have no fit value to compare against
        (["t1"], {"t1": []}),
        (["Q00"], {"Q00": {"t1": []}}),
        # lists must be flat
        (["t1"], {"t1": [[1e-6], [2e-6]]}),
        (["Q00"], {"Q00": {"t1": [[1e-6], [2e-6]]}}),
        # all elements of a list must be floats
        (["t1"], {"t1": [1e-6, "2e-6"]}),
        (["Q00"], {"Q00": {"t1": [1e-6, 2]}}),
    ],
)
def test_validate_metadata_invalid_expected_fit_result(var_names, expected_fit_result):
    ds = make_ds(var_names=var_names, expected_fit_result=expected_fit_result)
    with pytest.raises(ValueError, match="expected_fit_result"):
        validate_metadata(ds)


def test_validate_metadata_expected_fit_result_keys_mismatch():
    ds = make_ds(
        var_names=["Q00", "Q01"], expected_fit_result={"Q00": {"t1": [1e-6, 2e-6]}}
    )
    with pytest.raises(ValueError, match="each data variable"):
        validate_metadata(ds)
