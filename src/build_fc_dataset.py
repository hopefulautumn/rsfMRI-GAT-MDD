from __future__ import annotations

"""在 QC 之后构建可训练的 FC 数据集。

流程：
1. 按固定顺序读取被试（MDD -> HC），保证与 label.mat 对齐。
2. 对每个被试 ROISignals 做按 ROI 的 z-score。
3. 计算 Pearson 功能连接矩阵（ROI x ROI）。
4. 对相关矩阵做 Fisher Z 变换并将对角线置 0。
5. 保存为压缩 npz，供后续模型训练使用。
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


DATA_ROOT = Path("data")
OUTPUT_PATH = Path("processed/rest_meta_mdd_fc.npz")
# 读取顺序必须与标签编码约定一致。
# 当前数据集中 label.mat 的排列与 MDD->HC 对齐，因此保持此顺序不变。
GROUP_ORDER = ("MDD", "HC")


@dataclass(frozen=True)
class SubjectRecord:
    """单个被试文件的最小元信息。

    Attributes:
        subject_index: 全局样本索引（按读取顺序递增）。
        group_name: 来自路径的组名（MDD/HC）。
        mat_path: 对应 .mat 文件路径。
    """

    subject_index: int
    group_name: str
    site_id: str
    mat_path: Path


def extract_site_id(mat_path: Path) -> str:
    """从文件名中提取站点 ID（如 S01、S20）。"""

    m = re.match(r"^ROISignals_(S\d+)-\d+-\d+\.mat$", mat_path.name)
    if m is None:
        raise ValueError(f"无法从文件名解析站点 ID: {mat_path.name}")
    return m.group(1)


def load_labels(label_path: Path) -> np.ndarray:
    """读取标签并规范为一维 int64 向量。"""

    mat = loadmat(label_path)
    if "label" not in mat:
        raise KeyError(f"{label_path} 中未找到键 'label'")
    # 统一为 [N] 形状，避免后续 zip/索引时因列向量形状导致错位。
    return np.asarray(mat["label"]).reshape(-1).astype(np.int64)


def load_roi_timeseries(mat_path: Path) -> np.ndarray:
    """读取单被试 ROISignals，返回二维时间序列矩阵。"""

    mat = loadmat(mat_path)
    if "ROISignals" not in mat:
        raise KeyError(f"{mat_path} 中未找到键 'ROISignals'")
    # 用 float64 做 FC 计算更稳定，后续保存时仍可按需要降精度。
    ts = np.asarray(mat["ROISignals"], dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"{mat_path} 的 ROISignals 形状非法: {ts.shape}，应为二维矩阵")
    return ts


def collect_subjects(data_root: Path, group_order: tuple[str, str]) -> list[SubjectRecord]:
    """按固定分组顺序收集样本并建立稳定索引。"""

    subjects: list[SubjectRecord] = []
    idx = 0
    for group in group_order:
        # 文件名排序确保可复现：同一数据目录下每次运行顺序一致。
        for mat_path in sorted((data_root / group).glob("*.mat")):
            site_id = extract_site_id(mat_path)
            subjects.append(
                SubjectRecord(
                    subject_index=idx,
                    group_name=group,
                    site_id=site_id,
                    mat_path=mat_path,
                )
            )
            idx += 1
    return subjects


def zscore_by_roi(timeseries: np.ndarray) -> np.ndarray:
    """按 ROI 维度做 z-score 标准化。使每个 ROI 的均值为 0、标准差为 1

    输入 shape 通常为 [T, R]：
    - T: 时间点数
    - R: ROI 数
    """

    mean = timeseries.mean(axis=0, keepdims=True)
    std = timeseries.std(axis=0, keepdims=True)
    # 若某个 ROI 恒定，标准差为 0，避免除零。
    std = np.where(std == 0, 1.0, std)
    return (timeseries - mean) / std


def fisher_z(matrix: np.ndarray) -> np.ndarray:
    """对相关系数矩阵做 Fisher Z 变换。"""
    # 将取值范围从 [-1, 1] 映射到 (-∞, +∞)，并使变换后的值近似服从正态分布、方差稳定。
    # arctanh 在 ±1 处会发散，因此先裁剪到开区间附近。
    clipped = np.clip(matrix, -0.999999, 0.999999)
    return np.arctanh(clipped)


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    """从时间序列计算 FC 矩阵（Pearson + Fisher Z）。"""

    normalized = zscore_by_roi(timeseries)
    # 用于计算 Pearson 相关系数矩阵。rowvar=False 表示“列是变量（ROI），行是观测（时间点）”，结果为 [R, R]。
    fc = np.corrcoef(normalized, rowvar=False)
    # 对角线是 ROI 与自身相关，固定为 0 避免影响后续图构建。
    np.fill_diagonal(fc, 0.0)
    return fisher_z(fc)


def main() -> None:

    labels = load_labels(DATA_ROOT / "label.mat")
    subjects = collect_subjects(DATA_ROOT, GROUP_ORDER)

    if len(subjects) != len(labels):
        raise ValueError(
            f"样本文件数量 ({len(subjects)}) 与标签数量 ({len(labels)}) 不一致"
        )

    # 存储每个被试的 FC 矩阵（形状 [R, R]），后续会堆叠成三维数组。
    fc_matrices: list[np.ndarray] = []
    # 记录每个被试的组名字符串，便于调试或后续分析（如分层抽样）。
    group_names: list[str] = []
    # 记录每个被试的站点 ID（如 S01），用于后续分组交叉验证。
    site_ids: list[str] = []

    for subject in subjects:
        # 每个被试独立计算一个 [R, R] FC 矩阵。
        ts = load_roi_timeseries(subject.mat_path)
        fc = compute_fc(ts)
        fc_matrices.append(fc)
        group_names.append(subject.group_name)
        site_ids.append(subject.site_id)

    # 堆叠后形状为 [N, R, R]。
    fc_stack = np.stack(fc_matrices, axis=0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        # FC 主体数据。
        fc_matrices=fc_stack,
        # 与 fc_matrices 一一对应的标签（0/1）。
        labels=labels,
        # 记录每个样本从路径推断的组名，便于回溯。
        group_names=np.asarray(group_names, dtype=object),
        # 记录每个样本站点 ID，供分组交叉验证使用。
        site_ids=np.asarray(site_ids, dtype=object),
        # 显式保存读取顺序，避免后续脚本误用默认顺序。
        group_order=np.asarray(GROUP_ORDER, dtype=object),
    )

    unique, counts = np.unique(labels, return_counts=True)
    print(f"已保存文件: {OUTPUT_PATH}")
    print(f"样本数量: {fc_stack.shape[0]}")
    print(f"每个样本的 FC 形状: {fc_stack.shape[1:]}")
    print(f"标签分布: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"读取顺序: {GROUP_ORDER}")


if __name__ == "__main__":
    main()
