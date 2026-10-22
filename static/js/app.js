let predictionChart = null;

function renderPredictionResult(data) {
    const panel = document.getElementById("prediction-result");
    const title = document.getElementById("predicted-class");
    const chartArea = document.getElementById("probability-chart-area");
    const clusterArea = document.getElementById("cluster-result-area");
    const rawJson = document.getElementById("raw-json");
    const errorArea = document.getElementById("prediction-error");
    const badge = panel?.querySelector(".result-badge");

    if (!panel || !title || !chartArea || !rawJson) {
        console.error("Elemen hasil tidak ditemukan. Periksa module.html.");
        return;
    }

    panel.hidden = false;
    chartArea.hidden = true;

    if (clusterArea) clusterArea.hidden = true;

    if (errorArea) {
        errorArea.hidden = true;
        errorArea.textContent = "";
    }

    rawJson.textContent = JSON.stringify(data, null, 2);

    if (data.error) {
        title.textContent =
            typeof data.error === "string"
                ? data.error
                : "Prediksi gagal.";

        if (errorArea) {
            errorArea.textContent = title.textContent;
            errorArea.hidden = false;
        }

        if (badge) badge.textContent = "Perlu diperiksa";
        return;
    }

    // SUPERVISED LEARNING
    if (
        Array.isArray(data.top_probabilities) &&
        data.top_probabilities.length > 0
    ) {
        title.textContent = String(
            data.prediction || "Hasil Prediksi"
        ).replaceAll("_", " ");

        if (badge) badge.textContent = "Supervised Learning";

        chartArea.hidden = false;

        if (typeof Chart === "undefined") {
            if (errorArea) {
                errorArea.textContent =
                    "Chart.js belum dimuat. Periksa koneksi internet.";
                errorArea.hidden = false;
            }
            return;
        }

        const labels = data.top_probabilities.map(item =>
            String(item.class || "Kelas").replaceAll("_", " ")
        );

        const values = data.top_probabilities.map(item =>
            Number(item.probability) * 100
        );

        if (values.some(value => !Number.isFinite(value))) {
            if (errorArea) {
                errorArea.textContent =
                    "Format probabilitas dari API tidak valid.";
                errorArea.hidden = false;
            }
            return;
        }

        const canvas = document.getElementById("probability-chart");

        if (!canvas) {
            console.error("Canvas grafik tidak ditemukan.");
            return;
        }

        if (predictionChart) {
            predictionChart.destroy();
        }

        predictionChart = new Chart(canvas, {
            type: "bar",

            data: {
                labels: labels,
                datasets: [{
                    label: "Probabilitas",
                    data: values,
                    backgroundColor: [
                        "#356CE0",
                        "#36B9AE",
                        "#91A6C7",
                        "#A5B4CC",
                        "#CBD5E1"
                    ],
                    borderRadius: 7,
                    borderSkipped: false,
                    maxBarThickness: 38
                }]
            },

            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: false
                    },

                    tooltip: {
                        callbacks: {
                            label: context =>
                                context.parsed.x.toFixed(2) + "%"
                        }
                    }
                },

                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: "Probabilitas (%)"
                        },
                        ticks: {
                            callback: value => value + "%"
                        }
                    },

                    y: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        return;
    }

    // UNSUPERVISED LEARNING
    const clusterId = data.cluster_id ?? data.cluster;

    if (clusterId !== undefined && clusterId !== null) {
        title.textContent = "Cluster " + clusterId;

        if (badge) badge.textContent = "Unsupervised Learning";

        if (clusterArea) {
            clusterArea.hidden = false;

            document.getElementById("cluster-title").textContent =
                "Cluster " + clusterId;

            document.getElementById("cluster-profile").textContent =
                data.profile || data.description ||
                "Profil cluster belum tersedia.";
        }

        return;
    }

    title.textContent = data.message || "Respons API diterima.";
}


document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("prediction-form");

    if (!form) return;

    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        const button = form.querySelector('button[type="submit"]');
        const originalText = button ? button.textContent : "";

        if (button) {
            button.disabled = true;
            button.textContent = "Memproses...";
        }

        try {
            const kind = form.dataset.kind;
            const features = {};

            for (const [name, value] of new FormData(form).entries()) {
                const number = Number(value);

                if (value.trim() === "" || !Number.isFinite(number)) {
                    throw new Error(
                        `Nilai fitur ${name} harus berupa angka yang valid.`
                    );
                }

                features[name] = number;
            }

            const response = await fetch(
                `/api/predict/${encodeURIComponent(kind)}`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(features)
                }
            );

            let data;

            try {
                data = await response.json();
            } catch {
                throw new Error(
                    `Server tidak mengembalikan JSON yang valid (HTTP ${response.status}).`
                );
            }

            renderPredictionResult(data);

            if (!response.ok && !data.error) {
                const message = `Permintaan gagal (HTTP ${response.status}).`;
                const errorArea = document.getElementById("prediction-error");

                if (errorArea) {
                    errorArea.textContent = message;
                    errorArea.hidden = false;
                }
            }

        } catch (error) {
            renderPredictionResult({
                error: error.message ||
                    "Terjadi kesalahan saat menghubungi server."
            });

        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
    });

    form.addEventListener("reset", function () {
        const panel = document.getElementById("prediction-result");

        if (panel) panel.hidden = true;

        if (predictionChart) {
            predictionChart.destroy();
            predictionChart = null;
        }
    });
});