import pytest

from sqe_analysis.example_data import (
    get_dataset_names,
    open_dataset,
    validate_metadata,
)


def test_open_dataset_unknown_name():
    with pytest.raises(ValueError, match="not found"):
        open_dataset("nonexistent")


def test_all_metadata_valid():
    for ds_name in get_dataset_names():
        validate_metadata(open_dataset(ds_name))
