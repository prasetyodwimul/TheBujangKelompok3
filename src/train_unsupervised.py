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
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import ROOT, load_dataset


AREA_COLUMN = "Pixels_Areas"


def normalize_target(y):
    """Ubah target menjadi satu label kelas per baris."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            result = y.iloc[:, 0]
        else:
            # Untuk target one-hot dengan nama kelas sebagai kolom.
            result = y.idxmax(axis=1)

    elif isinstance(y, pd.Series):
        result = y

    else:
        arr = np.asarray(y)

        if arr.ndim == 2:
            if arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                arr = np.argmax(arr, axis=1)

        result = pd.Series(arr)

    return pd.Series(result).reset_index(drop=True)


def build_cluster_profiles(x, y, labels):
    """
    Membuat ringkasan tiap cluster setelah K-Means selesai.

    y hanya digunakan untuk interpretasi distribusi jenis cacat,
    tidak dipakai saat fitting atau pemilihan K.
    """
    labels = np.asarray(labels, dtype=int)
    y = normalize_target(y)

    if len(x) != len(y) or len(x) != len(labels):
        raise ValueError(
            "Jumlah baris fitur, target, dan label cluster tidak sama."
        )

    areas = None

    if AREA_COLUMN in x.columns:
        areas = pd.to_numeric(
            x[AREA_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)

    cluster_ids = sorted(np.unique(labels).tolist())

    # Median area setiap cluster untuk membandingkan ukuran cacat.
    median_by_cluster = {}

    if areas is not None:
        for cluster_id in cluster_ids:
            cluster_areas = areas[labels == cluster_id]
            valid_areas = cluster_areas[np.isfinite(cluster_areas)]

            median_by_cluster[cluster_id] = (
                float(np.median(valid_areas))
                if len(valid_areas)
                else None
            )

    target_values = y.fillna("Unknown").astype(str).to_numpy()

    profiles = {}

    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        member_count = int(mask.sum())

        # -----------------------------------------------------
        # Deskripsi relatif luas area
        # -----------------------------------------------------
        if areas is None:
            area_description = (
                "Pola luas area tidak tersedia karena fitur "
                f"{AREA_COLUMN} tidak ditemukan."
            )

        else:
            cluster_median = median_by_cluster[cluster_id]

            other_medians = [
                value
                for other_id, value in median_by_cluster.items()
                if other_id != cluster_id and value is not None
            ]

            if cluster_median is None:
                area_description = (
                    "Luas area belum dapat dibandingkan karena "
                    "nilainya tidak tersedia."
                )

            elif not other_medians:
                area_description = (
                    f"Median luas area {AREA_COLUMN} adalah "
                    f"{cluster_median:.2f}."
                )

            else:
                reference_area = float(np.mean(other_medians))

                if cluster_median > reference_area:
                    area_description = (
                        "Area cacat cenderung lebih besar"
                    )

                elif cluster_median < reference_area:
                    area_description = (
                        "Area cacat cenderung lebih kecil"
                    )

                else:
                    area_description = (
                        "Area cacat cenderung sebanding dengan "
                        "cluster lainnya"
                    )

        # -----------------------------------------------------
        # Distribusi kelas cacat setelah clustering
        # -----------------------------------------------------
        cluster_faults = pd.Series(
            target_values[mask],
            dtype=str,
        )

        counts = cluster_faults.value_counts()
        fault_distribution_percent = {}
        top_faults = []

        if len(counts):
            percentages = counts / counts.sum() * 100

            fault_distribution_percent = {
                str(fault_name): round(float(percent), 2)
                for fault_name, percent in percentages.items()
            }

            for fault_name, count in counts.head(3).items():
                top_faults.append({
                    "name": str(fault_name),
                    "count": int(count),
                    "percentage": round(
                        float(count / counts.sum() * 100),
                        2,
                    ),
                })

            dominant_fault = str(counts.index[0])
            dominant_percentage = float(
                counts.iloc[0] / counts.sum() * 100
            )

            if dominant_percentage >= 50:
                fault_description = (
                    f"didominasi {dominant_fault}"
                )

            elif len(counts) >= 2:
                first = str(counts.index[0])
                second = str(counts.index[1])

                fault_description = (
                    "dengan jenis cacat lebih beragam seperti "
                    f"{first} dan {second}"
                )

            else:
                fault_description = (
                    f"dengan jenis cacat utama {dominant_fault}"
                )

        else:
            dominant_fault = None
            dominant_percentage = None
            fault_description = (
                "distribusi jenis cacat tidak tersedia"
            )

        description = (
            f"{area_description}, {fault_description}."
        )

        # Statistik area yang ditampilkan di kartu profil.
        cluster_area_values = []

        if areas is not None:
            cluster_area_values = areas[mask]
            cluster_area_values = cluster_area_values[
                np.isfinite(cluster_area_values)
            ]

        mean_area = (
            float(np.mean(cluster_area_values))
            if len(cluster_area_values)
            else None
        )

        median_area = median_by_cluster.get(cluster_id)

        profiles[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "count": member_count,
            "mean_pixels_area": mean_area,
            "median_pixels_area": median_area,
            "dominant_fault": dominant_fault,
            "dominant_fault_percentage": dominant_percentage,
            "fault_distribution_percent": (
                fault_distribution_percent
            ),
            "top_faults": top_faults,
            "description": description,
        }

    return profiles


def save_cluster_comparison(
    x,
    y,
    labels,
    profiles,
    output_path,
):
    """
    Grafik perbandingan:
    1. Distribusi Pixels_Areas pada setiap cluster.
    2. Komposisi jenis cacat per cluster dalam persentase.
    """
    labels = np.asarray(labels, dtype=int)
    y = normalize_target(y).fillna("Unknown").astype(str)

    cluster_ids = sorted(np.unique(labels).tolist())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    # ---------------------------------------------------------
    # Grafik kiri: perbandingan luas area
    # ---------------------------------------------------------
    if AREA_COLUMN in x.columns:
        areas = pd.to_numeric(
            x[AREA_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)

        box_data = []
        box_labels = []

        for cluster_id in cluster_ids:
            values = areas[labels == cluster_id]
            values = values[np.isfinite(values)]

            if len(values):
                box_data.append(values)
                box_labels.append(f"Cluster {cluster_id}")

        if box_data:
            axes[0].boxplot(
                box_data,
                showfliers=False,
            )

            axes[0].set_xticks(
                range(1, len(box_labels) + 1)
            )
            axes[0].set_xticklabels(box_labels)
            axes[0].set_ylabel("Pixels_Areas")
            axes[0].set_title(
                "Perbandingan Luas Area Cacat"
            )
            axes[0].grid(axis="y", alpha=0.25)

        else:
            axes[0].text(
                0.5,
                0.5,
                "Data luas area tidak tersedia",
                ha="center",
                va="center",
            )
            axes[0].set_axis_off()

    else:
        axes[0].text(
            0.5,
            0.5,
            f"Fitur {AREA_COLUMN} tidak tersedia",
            ha="center",
            va="center",
        )
        axes[0].set_axis_off()

    # ---------------------------------------------------------
    # Grafik kanan: proporsi jenis cacat per cluster
    # ---------------------------------------------------------
    target_values = y.to_numpy(dtype=str)
    fault_names = sorted(np.unique(target_values).tolist())

    bottom = np.zeros(len(cluster_ids), dtype=float)
    cluster_names = [
        f"Cluster {cluster_id}"
        for cluster_id in cluster_ids
    ]

    for fault_name in fault_names:
        proportions = []

        for cluster_id in cluster_ids:
            member_mask = labels == cluster_id
            member_faults = target_values[member_mask]

            if len(member_faults):
                percent = (
                    np.sum(member_faults == fault_name)
                    / len(member_faults)
                    * 100
                )
            else:
                percent = 0.0

            proportions.append(float(percent))

        axes[1].bar(
            cluster_names,
            proportions,
            bottom=bottom,
            label=fault_name,
        )

        bottom += np.asarray(proportions)

    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Komposisi jenis cacat (%)")
    axes[1].set_title(
        "Komposisi Jenis Cacat per Cluster"
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(
        title="Jenis cacat",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.suptitle(
        "Perbandingan Karakteristik Cluster K-Means",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 0.94])

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def train():
    # y digunakan hanya untuk evaluasi dan interpretasi setelah
    # proses clustering, bukan untuk membentuk cluster.
    x, y = load_dataset()

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    y = normalize_target(y)

    if len(x) != len(y):
        raise ValueError(
            "Jumlah baris fitur dan target tidak cocok."
        )

    if len(x) < 3:
        raise ValueError(
            "Dataset terlalu kecil untuk membandingkan K-Means."
        )

    # ---------------------------------------------------------
    # Pilih K berdasarkan Silhouette Score
    # ---------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_imputed = imputer.fit_transform(x)
    x_scaled = scaler.fit_transform(x_imputed)

    candidates = []

    max_k = min(8, len(x) - 1)

    for k in range(2, max_k + 1):
        candidate = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels_candidate = candidate.fit_predict(x_scaled)

        unique_labels = np.unique(labels_candidate)

        if len(unique_labels) < 2 or len(unique_labels) >= len(x):
            continue

        score = silhouette_score(
            x_scaled,
            labels_candidate,
            sample_size=min(800, len(x)),
            random_state=42,
        )

        candidates.append({
            "k": int(k),
            "inertia": round(float(candidate.inertia_), 3),
            "silhouette": round(float(score), 4),
        })

    if not candidates:
        raise ValueError(
            "Tidak ditemukan kandidat K yang valid."
        )

    best_k = max(
        candidates,
        key=lambda item: item["silhouette"],
    )["k"]

    # ---------------------------------------------------------
    # Pipeline akhir
    # ---------------------------------------------------------
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "clustering",
            KMeans(
                n_clusters=best_k,
                random_state=42,
                n_init=10,
            ),
        ),
    ])

    labels = model.fit_predict(x)
    transformed = model[:-1].transform(x)

    # Profil dibuat setelah cluster sudah terbentuk.
    profiles = build_cluster_profiles(
        x=x,
        y=y,
        labels=labels,
    )

    y_for_evaluation = y.fillna("Unknown").astype(str)

    metrics = {
        "algorithm": "KMeans",
        "n_rows": int(len(x)),
        "n_features": int(x.shape[1]),
        "best_k": int(best_k),
        "candidates": candidates,
        "silhouette": round(
            float(
                silhouette_score(
                    transformed,
                    labels,
                    sample_size=min(800, len(x)),
                    random_state=42,
                )
            ),
            4,
        ),
        "cluster_counts": {
            str(cluster_id): int(
                np.sum(labels == cluster_id)
            )
            for cluster_id in sorted(np.unique(labels))
        },
        "cluster_profiles": profiles,
        "adjusted_rand_index_external": round(
            float(
                adjusted_rand_score(
                    y_for_evaluation,
                    labels,
                )
            ),
            4,
        ),
    }

    if 1 < len(np.unique(labels)) < len(x):
        metrics["davies_bouldin"] = round(
            float(
                davies_bouldin_score(
                    transformed,
                    labels,
                )
            ),
            4,
        )

        metrics["calinski_harabasz"] = round(
            float(
                calinski_harabasz_score(
                    transformed,
                    labels,
                )
            ),
            4,
        )

    # ---------------------------------------------------------
    # Simpan model, metrik, dan grafik
    # ---------------------------------------------------------
    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    images_dir = ROOT / "static" / "img"

    for folder in (
        models_dir,
        reports_dir,
        images_dir,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        model,
        models_dir / "unsupervised.joblib",
    )

    (reports_dir / "unsupervised_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Grafik pemilihan K.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].plot(
        [item["k"] for item in candidates],
        [item["inertia"] for item in candidates],
        "o-",
    )
    axes[0].set(
        xlabel="K",
        ylabel="Inertia (WCSS)",
        title="Elbow Method",
    )

    axes[1].plot(
        [item["k"] for item in candidates],
        [item["silhouette"] for item in candidates],
        "o-",
    )
    axes[1].set(
        xlabel="K",
        ylabel="Silhouette Score",
        title="Pemilihan K",
    )

    fig.tight_layout()
    fig.savefig(
        images_dir / "unsupervised_k_selection.png",
        dpi=160,
    )
    plt.close(fig)

    # Grafik PCA.
    pca = PCA(
        n_components=2,
        random_state=42,
    )
    points = pca.fit_transform(transformed)

    fig, ax = plt.subplots(figsize=(9, 6))

    chart = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="viridis",
        s=14,
        alpha=0.7,
    )

    fig.colorbar(
        chart,
        label="Cluster",
        ax=ax,
    )

    ax.set(
        title="PCA 2D — Kelompok Karakteristik Cacat",
        xlabel=(
            f"PC1 "
            f"({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        ),
        ylabel=(
            f"PC2 "
            f"({pca.explained_variance_ratio_[1] * 100:.1f}%)"
        ),
    )

    fig.tight_layout()
    fig.savefig(
        images_dir / "unsupervised_pca.png",
        dpi=160,
    )
    plt.close(fig)

    # Grafik perbandingan area dan komposisi jenis cacat.
    save_cluster_comparison(
        x=x,
        y=y,
        labels=labels,
        profiles=profiles,
        output_path=(
            images_dir / "unsupervised_cluster_comparison.png"
        ),
    )

    print("K optimal:", best_k)
    print("Silhouette:", metrics["silhouette"])

    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(profile["description"])
        print("Jumlah data:", profile["count"])

    print("\nModel disimpan ke models/unsupervised.joblib")
    print(
        "Grafik perbandingan disimpan ke "
        "static/img/unsupervised_cluster_comparison.png"
    )

    return model, metrics


if __name__ == "__main__":
    train()