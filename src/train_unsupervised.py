import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score,
    davies_bouldin_score, silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import ROOT, load_dataset


def train():
    # Label y hanya untuk pemeriksaan eksternal, TIDAK untuk membentuk cluster.
    x, y = load_dataset()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(x)
    x_scaled = scaler.fit_transform(x_imputed)

    candidates = []
    for k in range(2, 9):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = candidate.fit_predict(x_scaled)
        score = silhouette_score(
            x_scaled, labels,
            sample_size=min(800, len(x_scaled)), random_state=42,
        )
        candidates.append({
            "k": int(k),
            "inertia": round(float(candidate.inertia_), 3),
            "silhouette": round(float(score), 4),
        })

    best_k = max(candidates, key=lambda item: item["silhouette"])["k"]
    # Pipeline lengkap; backend akan memberikan DataFrame 27 fitur tanpa scaling manual.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clustering", KMeans(n_clusters=best_k, random_state=42, n_init=10)),
    ])
    labels = model.fit_predict(x)
    transformed = model[:-1].transform(x)

    metrics = {
        "algorithm": "KMeans",
        "n_rows": int(len(x)),
        "n_features": int(x.shape[1]),
        "best_k": int(best_k),
        "candidates": candidates,
        "silhouette": round(float(silhouette_score(
            transformed, labels, sample_size=min(800, len(x)), random_state=42
        )), 4),
        "davies_bouldin": round(float(davies_bouldin_score(transformed, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(transformed, labels)), 4),
        "cluster_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
        # Hanya indikator kesamaan dengan label; tidak berarti cluster = label cacat.
        "adjusted_rand_index_external": round(float(adjusted_rand_score(y, labels)), 4),
    }

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    images_dir = ROOT / "static" / "img"
    for folder in (models_dir, reports_dir, images_dir):
        folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "unsupervised.joblib")
    (reports_dir / "unsupervised_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([r["k"] for r in candidates],
                 [r["inertia"] for r in candidates], "o-")
    axes[0].set(xlabel="K", ylabel="Inertia (WCSS)", title="Elbow Method")
    axes[1].plot([r["k"] for r in candidates],
                 [r["silhouette"] for r in candidates], "o-")
    axes[1].set(xlabel="K", ylabel="Silhouette Score", title="Pemilihan K")
    fig.tight_layout()
    fig.savefig(images_dir / "unsupervised_k_selection.png", dpi=160)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(transformed)
    fig, ax = plt.subplots(figsize=(9, 6))
    chart = ax.scatter(points[:, 0], points[:, 1], c=labels,
                       cmap="viridis", s=14, alpha=0.7)
    fig.colorbar(chart, label="Cluster", ax=ax)
    ax.set(
        title="PCA 2D — Kelompok Karakteristik Cacat",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    )
    fig.tight_layout()
    fig.savefig(images_dir / "unsupervised_pca.png", dpi=160)
    plt.close(fig)

    print("K optimal (berdasarkan Silhouette):", best_k)
    print("Silhouette:", metrics["silhouette"])
    print("Davies-Bouldin:", metrics["davies_bouldin"])
    print("Calinski-Harabasz:", metrics["calinski_harabasz"])
    print("Adjusted Rand Index (hanya evaluasi eksternal):",
          metrics["adjusted_rand_index_external"])
    print("Model disimpan: models/unsupervised.joblib")
    return model, metrics


if __name__ == "__main__":
    train()