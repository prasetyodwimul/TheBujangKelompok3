(() => {
    "use strict";

    const form = document.getElementById("prediction-form");

    // Halaman lain tidak menggunakan form prediksi ini.
    if (!form) {
        return;
    }

    const kind = form.dataset.kind;
    const apiUrl = form.dataset.apiUrl;

    const submitButton = document.getElementById("submit-button");
    const statusElement = document.getElementById("prediction-status");
    const resultPanel = document.getElementById("result-panel");
    const resultTitle = document.getElementById("result-title");
    const resultSummary = document.getElementById("result-summary");
    const chartCanvas = document.getElementById("prediction-chart");

    let predictionChart = null;

    function setStatus(message, isError = false) {
        if (!statusElement) {
            return;
        }

        statusElement.textContent = message;
        statusElement.style.color = isError ? "#d83d3d" : "inherit";
    }

    function destroyChart() {
        if (predictionChart) {
            predictionChart.destroy();
            predictionChart = null;
        }
    }

    function renderSupervised(data) {
        if (resultTitle) {
            resultTitle.textContent = `Prediksi: ${data.prediction}`;
        }

        const probabilities = data.top_probabilities || [];

        if (resultSummary) {
            resultSummary.textContent = probabilities
                .map(item => {
                    const percent = (
                        Number(item.probability) * 100
                    ).toFixed(1);

                    return `${item.class}: ${percent}%`;
                })
                .join(" | ");
        }

        destroyChart();

        if (!chartCanvas || !window.Chart || !probabilities.length) {
            return;
        }

        predictionChart = new Chart(chartCanvas, {
            type: "bar",
            data: {
                labels: probabilities.map(item => item.class),
                datasets: [{
                    label: "Probabilitas (%)",
                    data: probabilities.map(
                        item => Number(item.probability) * 100
                    )
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: "Probabilitas (%)"
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    function renderUnsupervised(data) {
        const clusterId = Number(data.cluster_id ?? data.cluster);

        if (!Number.isInteger(clusterId) || clusterId < 0) {
            throw new Error("ID cluster dari model tidak valid.");
        }

        if (resultTitle) {
            resultTitle.textContent = `Hasil Segmentasi: Cluster ${clusterId}`;
        }

        if (resultSummary) {
            resultSummary.textContent =
                data.profile ||
                `Input dikelompokkan ke Cluster ${clusterId}. ` +
                "Profil tersedia pada ringkasan cluster di bawah.";
        }

        destroyChart();
    }

    form.addEventListener("submit", async event => {
        event.preventDefault();

        if (!form.reportValidity()) {
            return;
        }

        const originalButtonText = submitButton
            ? submitButton.textContent
            : "";

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = "Memproses...";
        }

        if (resultPanel) {
            resultPanel.hidden = true;
        }

        setStatus("Sedang memproses input...");

        try {
            const formData = new FormData(form);
            const payload = {};

            for (const [key, value] of formData.entries()) {
                payload[key] = value;
            }

            const response = await fetch(apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    `Permintaan gagal dengan status ${response.status}.`
                );
            }

            if (kind === "supervised") {
                renderSupervised(data);
                setStatus("Prediksi berhasil.");

            } else if (kind === "unsupervised") {
                renderUnsupervised(data);
                setStatus("Clustering berhasil.");

            } else {
                throw new Error("Jenis modul tidak dikenal.");
            }

            if (resultPanel) {
                resultPanel.hidden = false;
                resultPanel.scrollIntoView({
                    behavior: "smooth",
                    block: "nearest"
                });
            }

        } catch (error) {
            console.error("Prediksi gagal:", error);

            setStatus(
                error.message || "Terjadi kesalahan.",
                true
            );

        } finally {
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = originalButtonText;
            }
        }
    });
})();