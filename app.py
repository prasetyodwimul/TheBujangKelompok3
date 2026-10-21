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
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def schema():
    return read_json(ROOT / "reports" / "schema.json")


def model_path(kind):
    names = {
        "supervised": "supervised.joblib",
        "unsupervised": "unsupervised.joblib",
    }
    return ROOT / "models" / names[kind]


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
        "module.html", kind=kind, features=sch.get("features", []),
        medians=sch.get("medians", {}), ready=model_path(kind).exists(),
    )


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "schema_ready": bool(schema().get("features")),
        "supervised_ready": model_path("supervised").exists(),
        "unsupervised_ready": model_path("unsupervised").exists(),
    })


@app.get("/api/metrics")
def metrics():
    return jsonify({
        "supervised": read_json(ROOT / "reports" / "supervised_metrics.json"),
        "unsupervised": read_json(ROOT / "reports" / "unsupervised_metrics.json"),
    })


@app.post("/api/predict/<kind>")
def predict(kind):
    if kind not in ("supervised", "unsupervised"):
        return jsonify({"error": "Modul tidak ditemukan"}), 404
    sch = schema()
    features = sch.get("features", [])
    if not features:
        return jsonify({"error": "Schema data belum disiapkan Orang 2"}), 503
    path = model_path(kind)
    if not path.exists():
        return jsonify({"error": f"Model {kind} belum dilatih"}), 503

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not isinstance(payload, dict):
        return jsonify({"error": "Input harus berupa object JSON atau form"}), 400
    values = []
    try:
        for name in features:
            value = float(payload[name])
            if not np.isfinite(value):
                raise ValueError(f"{name} bukan bilangan hingga")
            values.append(value)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Input tidak valid: {exc}"}), 400

    row = pd.DataFrame([values], columns=features)
    # Hanya model hasil training milik tim sendiri boleh dibaca dari folder ini.
    model = joblib.load(path)
    result = model.predict(row)[0]
    if kind == "supervised":
        probabilities = model.predict_proba(row)[0]
        ranks = sorted(
            zip(model.classes_.tolist(), probabilities.tolist()),
            key=lambda item: item[1], reverse=True,
        )[:3]
        return jsonify({
            "module": "supervised", "prediction": str(result),
            "top_probabilities": [
                {"class": str(label), "probability": round(float(prob), 4)}
                for label, prob in ranks
            ],
        })
    return jsonify({"module": "unsupervised", "cluster": int(result)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)