from pathlib import Path
import pytest

from src.data_pipeline import ROOT, load_dataset


def test_unsupervised_prediction_if_artifact_exists():
    path = ROOT / "models" / "unsupervised.joblib"
    if not path.exists() or not (ROOT / "data/raw/faults.csv").exists():
        pytest.skip("Artefak/dataset lokal belum tersedia")
    import joblib
    x, y = load_dataset()
    model = joblib.load(path)
    cluster_id = model.predict(x.iloc[[0]])[0]
    assert 0 <= int(cluster_id) < model.named_steps["clustering"].n_clusters
    assert len(model.feature_names_in_) == 27