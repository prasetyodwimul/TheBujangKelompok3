from pathlib import Path
import pytest

from src.data_pipeline import FEATURES, FAULT_TYPES, RAW, load_dataset


def test_contract_features_and_labels():
    assert len(FEATURES) == 27
    assert len(FAULT_TYPES) == 7
    assert not set(FEATURES).intersection(FAULT_TYPES)


def test_actual_dataset_if_available():
    if not Path(RAW).exists():
        pytest.skip("faults.csv tidak ada; unduh sesuai panduan Orang 2")
    x, y = load_dataset()
    assert len(x) == len(y)
    assert list(x.columns) == FEATURES
    assert y.isna().sum() == 0