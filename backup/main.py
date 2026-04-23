from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


DATA_ROOT = Path("data")


def load_roi_timeseries(mat_path: Path) -> np.ndarray:
    """Load one subject's ROI time series matrix from a MAT file."""
    mat = loadmat(mat_path)
    if "ROISignals" not in mat:
        raise KeyError(f"{mat_path} does not contain key 'ROISignals'")
    return np.asarray(mat["ROISignals"], dtype=np.float64)


def load_labels(label_path: Path) -> np.ndarray:
    """Load labels from label.mat and flatten to shape (N,)."""
    mat = loadmat(label_path)
    if "label" not in mat:
        raise KeyError(f"{label_path} does not contain key 'label'")
    return np.asarray(mat["label"]).reshape(-1)


def inspect_mat_file(mat_path: Path) -> None:
    """Print keys, shapes, dtypes, and a small preview of a MAT file."""
    mat = loadmat(mat_path)
    keys = [key for key in mat.keys() if not key.startswith("__")]
    print(f"File: {mat_path}")
    print(f"Keys: {keys}")
    for key in keys:
        value = np.asarray(mat[key])
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        flat = value.reshape(-1)
        preview = flat[:10]
        print(f"  {key} preview: {preview}")


def main() -> None:
    if len(sys.argv) > 1:
        inspect_mat_file(Path(sys.argv[1]))
        return

    hc_example = DATA_ROOT / "HC" / "ROISignals_S01-2-0001.mat"
    roi_ts = load_roi_timeseries(hc_example)
    print(f"ROI time series shape: {roi_ts.shape}")
    print("Expected meaning: (time_points, n_rois)")

    labels = load_labels(DATA_ROOT / "label.mat")
    unique, counts = np.unique(labels, return_counts=True)
    label_stats = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Label shape: {labels.shape}")
    print(f"Label distribution: {label_stats}")


if __name__ == "__main__":
    main()
