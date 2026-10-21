from pathlib import Path

import pandas as pd

FAULT_COLUMNS = [
    'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
    'Dirtiness', 'Bumps', 'Other_Faults'
]


def load_faults(path='data/raw/faults.csv'):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f'Dataset tidak ditemukan: {csv_path}')

    df = pd.read_csv(csv_path)
    missing_labels = [name for name in FAULT_COLUMNS if name not in df.columns]
    if missing_labels:
        raise ValueError(f'Kolom target tidak ditemukan: {missing_labels}')

    if not df[FAULT_COLUMNS].isin([0, 1]).all().all():
        raise ValueError('Target harus berupa angka biner 0/1.')

    if not df[FAULT_COLUMNS].sum(axis=1).eq(1).all():
        raise ValueError('Setiap observasi harus memiliki tepat satu jenis cacat.')

    X = df.drop(columns=FAULT_COLUMNS).apply(pd.to_numeric, errors='coerce')
    if X.shape[1] != 27:
        raise ValueError(f'Jumlah fitur tidak sesuai: {X.shape[1]} (harus 27).')
    if X.isna().all().any():
        raise ValueError('Ada fitur yang seluruh nilainya kosong/tidak numerik.')

    y = df[FAULT_COLUMNS].idxmax(axis=1)
    return X, y