from __future__ import annotations

"""rs-fMRI ROI 时间序列数据概览脚本。

本脚本完成两件事：
1. 读取标签与被试文件，检查数量是否一致。
2. 导出一份 CSV：
    - timeseries_overview.csv：样本基础信息报告。

已知数据集顺序约定：
- 按目录读取顺序固定为 MDD -> HC。
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATA_ROOT = Path("data")
OUTPUT_DIR = Path("processed")
# 固定读取顺序：先 MDD 后 HC。
# 这是基于当前数据集的 label 序列结构确定的。
GROUP_ORDER = ("MDD", "HC")


@dataclass(frozen=True)
class SubjectRecord:
    """单个被试文件的元信息。"""

    # 按当前读取顺序生成的连续索引（从 0 开始）。
    subject_index: int
    # 从路径中得到的分组名（HC 或 MDD）。
    group_name: str
    # 对应 .mat 文件路径。
    mat_path: Path


def load_labels(label_path: Path) -> np.ndarray:
    """读取 label.mat 中的 label 并展平成一维数组。"""

    mat = loadmat(label_path)
    if "label" not in mat:
        raise KeyError(f"{label_path} does not contain key 'label'")
    return np.asarray(mat["label"]).reshape(-1)


def load_roi_timeseries(mat_path: Path) -> np.ndarray:
    """读取单个被试的 ROISignals 时间序列矩阵。"""

    mat = loadmat(mat_path)
    if "ROISignals" not in mat:
        raise KeyError(f"{mat_path} does not contain key 'ROISignals'")
    return np.asarray(mat["ROISignals"], dtype=np.float64)


def collect_subjects(data_root: Path, group_order: tuple[str, str]) -> list[SubjectRecord]:
    """按给定分组顺序收集所有被试文件并建立索引。

    例如 group_order=("MDD", "HC") 时，先遍历 data/MDD，再遍历 data/HC。
    """

    subjects: list[SubjectRecord] = []
    idx = 0
    for group in group_order:
        # 文件名排序后再读取，保证顺序稳定可复现。
        for mat_path in sorted((data_root / group).glob("*.mat")):
            subjects.append(SubjectRecord(subject_index=idx, group_name=group, mat_path=mat_path))
            idx += 1
    return subjects


def write_timeseries_overview(
    subjects: list[SubjectRecord],
    labels: np.ndarray,
    output_dir: Path,
) -> Path:
    """导出逐被试基础信息报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "timeseries_overview.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject_index",
                "group_from_path",
                "mat_file",
                "time_points",
                "n_rois",
                "label",
            ]
        )

        for subject, label in zip(subjects, labels):
            ts = load_roi_timeseries(subject.mat_path)
            # 合法数据应为二维矩阵：(time_points, n_rois)。
            if ts.ndim != 2:
                print("存在非二维 ROISignals 数据，无法读取时间点和 ROI 数量。")
                exit(1)

            time_points, n_rois = ts.shape

            writer.writerow(
                [
                    subject.subject_index,
                    subject.group_name,
                    str(subject.mat_path),
                    time_points,
                    n_rois,
                    int(label),
                ]
            )

    return out_path


def print_summary(subjects: list[SubjectRecord], labels: np.ndarray, order_name: str) -> None:
    """打印数据集概览（仅数量检查）。"""

    groups = [s.group_name for s in subjects]
    hc_count = sum(g == "HC" for g in groups)
    mdd_count = sum(g == "MDD" for g in groups)
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"分组读取顺序: {order_name}")
    print(f"样本总数: {len(subjects)}")
    print(f"HC 数量（按路径）: {hc_count}")
    print(f"MDD 数量（按路径）: {mdd_count}")
    print(f"标签分布: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

def main() -> None:

    # 读取标签序列。
    labels = load_labels(DATA_ROOT / "label.mat")

    # 按固定顺序收集样本，确保与当前数据集约定一致。
    subjects = collect_subjects(DATA_ROOT, GROUP_ORDER)
    if len(subjects) != len(labels):
        print(f"Subject count: {len(subjects)}, Label count: {len(labels)}")
        exit(1)

    # 打印摘要并导出报告。
    print_summary(subjects, labels, order_name=",".join(GROUP_ORDER))
    overview_path = write_timeseries_overview(subjects, labels, OUTPUT_DIR)

    print(f"Saved timeseries overview: {overview_path}")


if __name__ == "__main__":
    main()
