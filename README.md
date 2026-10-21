# The Bujang Kelompok 3 — Steel Plates Faults

Aplikasi web Machine Learning yang menggabungkan:
- **Supervised:** Random Forest untuk klasifikasi 7 jenis cacat pelat baja.
- **Unsupervised:** K-Means untuk eksplorasi kelompok karakteristik cacat.

Dataset: [Kaggle Faulty Steel Plates](https://www.kaggle.com/datasets/uciml/faulty-steel-plates) / [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults).

**Lisensi sumber UCI:** CC BY 4.0; atribusi: Buscema, Terzi, dan Tastle, *Steel Plates Faults* (2010), UCI Machine Learning Repository, DOI: 10.24432/C5J88N.

## Menjalankan

1. Buka terminal PowerShell pada folder repo.
2. `py -3.11 -m venv .venv`
3. `.\.venv\Scripts\Activate.ps1`
4. `python -m pip install -r requirements.txt`
5. Letakkan dataset CSV sebagai `data/raw/faults.csv` (file mentah tidak dilacak Git).
6. `python -m src.data_pipeline` (setelah tahap Orang 2).
7. `python -m src.train_supervised` (setelah tahap Orang 3).
8. `python -m src.train_unsupervised` (setelah tahap Orang 4).
9. `python app.py` dan buka <http://127.0.0.1:5000>.

## Panduan kerja

Lihat `docs/00_PANDUAN_UTAMA.md`, kemudian selesaikan Orang 1 → Orang 2 → Orang 3 → Orang 4 secara berurutan. Semua fitur masuk **`dev` melalui Pull Request**; `main` baru diisi/diperbarui setelah integrasi final selesai.
```