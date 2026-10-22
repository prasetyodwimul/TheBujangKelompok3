from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "faults.csv"

# Membaca konfigurasi yang dihasilkan notebook EDA

SCHEMA_PATH = ROOT / "reports" / "schema.json"

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(
        "schema.json tidak ditemukan. "
        "Jalankan notebook EDA sampai Cell 13."
    )

with SCHEMA_PATH.open(encoding="utf-8") as file:
    schema_config = json.load(file)

FEATURES = schema_config["features"]
FAULT_TYPES = schema_config["fault_types"]

assert len(FEATURES) == 27
assert len(FAULT_TYPES) == 7


def load_dataset(path=RAW):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Unduh dari Kaggle."
        )
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()

    # Sebagian salinan menggunakan ejaan K_Scratch.
    if "K_Scratch" in frame.columns and "K_Scatch" not in frame.columns:
        frame = frame.rename(columns={"K_Scratch": "K_Scatch"})

    required = FEATURES + FAULT_TYPES
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak sesuai; hilang: {missing}")

    x = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if x.isna().all(axis=0).any():
        raise ValueError("Ada fitur yang seluruh nilainya kosong atau non-numerik")

    indicators = frame[FAULT_TYPES].apply(pd.to_numeric, errors="coerce")
    valid = indicators.isin([0, 1]).all(axis=1) & (indicators.sum(axis=1) == 1)
    if not valid.all():
        raise ValueError(
            f"Ditemukan {(~valid).sum()} baris dengan label tidak valid; "
            "setiap baris harus memiliki tepat satu label aktif."
        )

    y = indicators.idxmax(axis=1).astype(str)
    return x, y


def save_outputs():
    x, y = load_dataset()
    report_dir = ROOT / "reports"
    chart_dir = ROOT / "static" / "img"
    processed_dir = ROOT / "data" / "processed"
    for folder in (report_dir, chart_dir, processed_dir):
        folder.mkdir(parents=True, exist_ok=True)

    medians = x.median(numeric_only=True)
    if medians.isna().any():
        raise ValueError("Median suatu fitur tidak dapat dihitung")

    schema = {
        "features": FEATURES,
        "fault_types": FAULT_TYPES,
        "medians": {col: float(medians[col]) for col in FEATURES},
        "source": "UCI Steel Plates Faults / Kaggle uciml/faulty-steel-plates",
        "n_rows": int(len(x)),
    }
    summary = {
        "rows": int(len(x)),
        "n_features": len(FEATURES),
        "missing_by_feature": {k: int(v) for k, v in x.isna().sum().items()},
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
    }
    (report_dir / "schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    (report_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # File processed lokal tidak akan di-push karena *.csv diabaikan.
    x.to_csv(processed_dir / "steel_features.csv", index=False)
    y.to_frame(name="Fault_Type").to_csv(
        processed_dir / "steel_labels.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, column in zip(axes.flat, [
        "Pixels_Areas", "Steel_Plate_Thickness", "Edges_Index", "Luminosity_Index"
    ]):
        ax.hist(x[column].dropna(), bins=24, color="#376c96", edgecolor="white")
        ax.set_title(column)
    fig.tight_layout()
    fig.savefig(chart_dir / "eda_distribusi.png", dpi=150)
    plt.close(fig)
    print("Ukuran fitur:", x.shape)
    print("Jenis kelas:", y.value_counts().to_dict())
    print("Total missing:", int(x.isna().sum().sum()))
    print("Hasil: reports/schema.json, reports/eda_summary.json")


if __name__ == "__main__":
    save_outputs()