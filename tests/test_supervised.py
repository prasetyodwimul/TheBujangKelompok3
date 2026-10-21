from pathlib import Path
import pytest

from src.data_pipeline import ROOT, load_dataset


def test_supervised_prediction_if_artifact_exists():
    path = ROOT / "models" / "supervised.joblib"
    if not path.exists() or not (ROOT / "data/raw/faults.csv").exists():
        pytest.skip("Artefak/dataset lokal belum tersedia")
    import joblib
    x, y = load_dataset()
    model = joblib.load(path)
    assert model.predict(x.iloc[[0]])[0] in set(y.unique())
    assert len(model.classes_) == 7