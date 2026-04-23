from __future__ import annotations

"""打印单个被试时间序列的局部切片。

默认打印：前 10 行（时间点）x 前 8 列（ROI）。
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


# 按需改成其他文件，例如 HC 样本。
MAT_PATH = Path("data/MDD/ROISignals_S01-1-0001.mat")
ROW_COUNT = 10
COL_COUNT = 8


def main() -> None:
    # 基础存在性检查，避免路径写错时直接崩溃在 loadmat。
    if not MAT_PATH.exists():
        raise FileNotFoundError(f"未找到文件: {MAT_PATH}")

    # 读取 .mat，并确认包含我们需要的 ROISignals 键。
    mat = loadmat(MAT_PATH)
    if "ROISignals" not in mat:
        raise KeyError(f"{MAT_PATH} 不包含键 'ROISignals'")

    # 转为 numpy 二维矩阵，约定 shape=(时间点, ROI)。
    ts = np.asarray(mat["ROISignals"], dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"ROISignals 不是二维矩阵: {ts.shape}")

    # 防止越界：如果请求行/列超过实际大小，则自动截到最大可用范围。
    # 同时保证最少打印 1 行 1 列，避免传入 0 或负数造成空切片。
    r = max(1, min(ROW_COUNT, ts.shape[0]))
    c = max(1, min(COL_COUNT, ts.shape[1]))

    # 打印元信息与切片内容，便于快速人工查看原始时间序列数值。
    print(f"文件: {MAT_PATH}")
    print(f"整体形状: {ts.shape} (行=时间点, 列=ROI)")
    print(f"打印切片: 前 {r} 行 x 前 {c} 列")
    print(np.array2string(ts[:r, :c], precision=4, suppress_small=False))


if __name__ == "__main__":
    main()
