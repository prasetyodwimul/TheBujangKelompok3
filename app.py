from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)


def read_json(path):
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        app.logger.exception("Gagal membaca JSON: %s", path)
        return {}


def schema():
    return read_json(ROOT / "reports" / "schema.json")


def model_path(kind):
    names = {
        "supervised": "supervised.joblib",
        "unsupervised": "unsupervised.joblib",
    }

    if kind not in names:
        raise ValueError("Modul tidak dikenal.")

    return ROOT / "models" / names[kind]


@lru_cache(maxsize=2)
def load_model(kind):
    path = model_path(kind)

    if not path.exists():
        raise FileNotFoundError(
            f"Model {kind} tidak ditemukan."
        )

    return joblib.load(path)


def get_model_features(model, schema_features):
    model_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_names is None:
        return list(schema_features)

    model_features = [str(name) for name in model_names]
    schema_features = [str(name) for name in schema_features]

    if set(model_features) != set(schema_features):
        raise RuntimeError(
            "Fitur pada schema.json berbeda dengan fitur model. "
            "Periksa kembali nama kolom."
        )

    return model_features


def get_cluster_profiles():
    metrics_data = read_json(
        ROOT / "reports" / "unsupervised_metrics.json"
    )

    profiles = metrics_data.get("cluster_profiles", {})

    return profiles if isinstance(profiles, dict) else {}


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/module/<kind>")
def module_page(kind):
    if kind not in ("supervised", "unsupervised"):
        return "Modul tidak ditemukan", 404

    sch = schema()

    return render_template(
        "module.html",
        kind=kind,
        features=sch.get("features", []),
        medians=sch.get("medians", {}),
        ready=model_path(kind).exists(),
        cluster_profiles=(
            get_cluster_profiles()
            if kind == "unsupervised"
            else {}
        ),
        comparison_ready=(
            ROOT
            / "static"
            / "img"
            / "unsupervised_cluster_comparison.png"
        ).exists(),
        pca_ready=(
            ROOT / "static" / "img" / "unsupervised_pca.png"
        ).exists(),
    )


@app.get("/api/health")
def health():
    sch = schema()

    return jsonify({
        "status": "ok",
        "schema_ready": bool(sch.get("features")),
        "supervised_ready": model_path("supervised").exists(),
        "unsupervised_ready": model_path("unsupervised").exists(),
    })


@app.get("/api/metrics")
def metrics():
    return jsonify({
        "supervised": read_json(
            ROOT / "reports" / "supervised_metrics.json"
        ),
        "unsupervised": read_json(
            ROOT / "reports" / "unsupervised_metrics.json"
        ),
    })


@app.post("/api/predict/<kind>")
def predict(kind):
    if kind not in ("supervised", "unsupervised"):
        return jsonify({
            "error": "Modul tidak ditemukan."
        }), 404

    sch = schema()
    features = sch.get("features", [])

    if not features:
        return jsonify({
            "error": "Schema data belum disiapkan Orang 2."
        }), 503

    if not model_path(kind).exists():
        return jsonify({
            "error": f"Model {kind} belum dilatih."
        }), 503

    payload = (
        request.get_json(silent=True)
        if request.is_json
        else request.form.to_dict()
    )

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Input harus berupa object JSON atau form."
        }), 400

    try:
        model = load_model(kind)
        model_features = get_model_features(model, features)

        values = {}

        for name in features:
            if name not in payload:
                raise ValueError(
                    f"Fitur {name} belum diisi."
                )

            raw_value = payload[name]

            if raw_value is None or str(raw_value).strip() == "":
                raise ValueError(
                    f"Fitur {name} belum diisi."
                )

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Nilai {name} harus berupa angka."
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"Nilai {name} tidak valid."
                )

            values[name] = value

        # Jaga urutan kolom agar sesuai dengan saat model dilatih.
        row = pd.DataFrame(
            [[values[name] for name in model_features]],
            columns=model_features,
        )

        result = model.predict(row)[0]

        # -----------------------------------------------------
        # SUPERVISED
        # -----------------------------------------------------
        if kind == "supervised":
            if not hasattr(model, "predict_proba"):
                return jsonify({
                    "error": (
                        "Model supervised tidak mendukung "
                        "predict_proba."
                    )
                }), 500

            probabilities = model.predict_proba(row)[0]

            ranks = sorted(
                zip(
                    model.classes_.tolist(),
                    probabilities.tolist(),
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:3]

            return jsonify({
                "module": "supervised",
                "prediction": str(result),
                "top_probabilities": [
                    {
                        "class": str(label),
                        "probability": round(
                            float(probability),
                            4,
                        ),
                    }
                    for label, probability in ranks
                ],
            })

        # -----------------------------------------------------
        # UNSUPERVISED
        # -----------------------------------------------------
        cluster_id = int(result)

        profiles = get_cluster_profiles()
        cluster_profile = profiles.get(str(cluster_id), {})

        profile_text = cluster_profile.get(
            "description",
            (
                f"Input dikelompokkan ke Cluster {cluster_id}. "
                "Deskripsi cluster belum tersedia. "
                "Jalankan ulang training unsupervised."
            ),
        )

        return jsonify({
            "module": "unsupervised",
            "cluster": cluster_id,
            "cluster_id": cluster_id,
            "prediction": f"Cluster {cluster_id}",
            "profile": profile_text,
            "profile_details": cluster_profile,
        })

    except ValueError as error:
        return jsonify({
            "error": f"Input tidak valid: {error}"
        }), 400

    except FileNotFoundError:
        app.logger.exception("Model tidak ditemukan.")

        return jsonify({
            "error": f"Model {kind} belum tersedia."
        }), 503

    except RuntimeError as error:
        app.logger.exception("Konfigurasi model tidak cocok.")

        return jsonify({
            "error": str(error)
        }), 503

    except Exception:
        app.logger.exception("Prediksi %s gagal.", kind)

        return jsonify({
            "error": f"Gagal menjalankan prediksi {kind}."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )