from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import ROOT, FEATURES, FAULT_TYPES, load_dataset


def train():
    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    # Imputer dan scaler ada DI DALAM pipeline agar hanya fit di training.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1,
        )),
    ])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    report = classification_report(
        y_test, predicted, labels=FAULT_TYPES,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "algorithm": "RandomForestClassifier",
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "classes": FAULT_TYPES,
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "f1_macro": round(float(f1_score(
            y_test, predicted, average="macro", zero_division=0
        )), 4),
        "classification_report": report,
    }

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    images_dir = ROOT / "static" / "img"
    for folder in (models_dir, reports_dir, images_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # Pipeline lengkap: aman untuk inference dengan DataFrame berisi 27 fitur.
    joblib.dump(model, models_dir / "supervised.joblib")
    (reports_dir / "supervised_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test, predicted, labels=FAULT_TYPES)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=FAULT_TYPES, yticklabels=FAULT_TYPES, ax=ax)
    ax.set(xlabel="Prediksi", ylabel="Aktual", title="Confusion Matrix — Random Forest")
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(images_dir / "supervised_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("Model supervised disimpan: models/supervised.joblib")
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["f1_macro"])
    print(classification_report(
        y_test, predicted, labels=FAULT_TYPES, zero_division=0
    ))
    return model, metrics


if __name__ == "__main__":
    train()