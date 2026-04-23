from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


DATA_ROOT = Path('data')
HC_DIR = DATA_ROOT / 'HC'
MDD_DIR = DATA_ROOT / 'MDD'
LABEL_PATH = DATA_ROOT / 'label.mat'
OUTPUT_DIR = Path('processed')


@dataclass(frozen=True)
class SubjectRecord:
    path: Path
    group_name: str


def load_roi_timeseries(mat_path: Path) -> np.ndarray:
    mat = loadmat(mat_path)
    if 'ROISignals' not in mat:
        raise KeyError(f"{mat_path} does not contain key 'ROISignals'")
    return np.asarray(mat['ROISignals'], dtype=np.float64)


def load_labels(label_path: Path) -> np.ndarray:
    mat = loadmat(label_path)
    if 'label' not in mat:
        raise KeyError(f"{label_path} does not contain key 'label'")
    return np.asarray(mat['label']).reshape(-1)


def zscore_by_roi(timeseries: np.ndarray) -> np.ndarray:
    mean = timeseries.mean(axis=0, keepdims=True)
    std = timeseries.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (timeseries - mean) / std


def fisher_z(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(matrix, -0.999999, 0.999999)
    return np.arctanh(clipped)


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    normalized = zscore_by_roi(timeseries)
    fc = np.corrcoef(normalized, rowvar=False)
    np.fill_diagonal(fc, 0.0)
    return fisher_z(fc)


def collect_subject_files(root: Path) -> list[SubjectRecord]:
    records: list[SubjectRecord] = []
    for group_name in ('HC', 'MDD'):
        group_dir = root / group_name
        for mat_path in sorted(group_dir.glob('*.mat')):
            records.append(SubjectRecord(path=mat_path, group_name=group_name))
    return records


def preprocess_dataset(root: Path = DATA_ROOT) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = collect_subject_files(root)
    labels = load_labels(root / 'label.mat')
    if len(records) != len(labels):
        raise ValueError(
            f'Subject file count ({len(records)}) does not match label count ({len(labels)})'
        )

    features: list[np.ndarray] = []
    group_names: list[str] = []
    for record in records:
        timeseries = load_roi_timeseries(record.path)
        features.append(compute_fc(timeseries))
        group_names.append(record.group_name)

    return np.stack(features, axis=0), labels.astype(np.int64), group_names


def save_preprocessed_data(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fc_matrices, labels, group_names = preprocess_dataset()
    output_path = output_dir / 'rest_meta_mdd_fc.npz'
    np.savez_compressed(
        output_path,
        fc_matrices=fc_matrices,
        labels=labels,
        group_names=np.asarray(group_names, dtype=object),
    )
    return output_path


def print_summary() -> None:
    records = collect_subject_files(DATA_ROOT)
    labels = load_labels(LABEL_PATH)
    print(f'Subject files: {len(records)}')
    print(f'Labels: {labels.shape[0]}')
    print(f'HC files: {sum(record.group_name == "HC" for record in records)}')
    print(f'MDD files: {sum(record.group_name == "MDD" for record in records)}')

    example = load_roi_timeseries(records[0].path)
    print(f'Example file: {records[0].path}')
    print(f'ROI timeseries shape: {example.shape}')
    print(f'FC matrix shape: {compute_fc(example).shape}')


if __name__ == '__main__':
    print_summary()
    output = save_preprocessed_data()
    print(f'Saved: {output}')
