# 读取并展示 label.mat，同时导出 label.csv 的简化脚本。
# 显示标签结构与分布，便于快速核对。导出 index,label 两列 CSV 供后续脚本使用。

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


MAT_PATH = Path("data/label.mat")
OUTPUT_CSV = Path("output/label.csv")

def main() -> None:
	# 先检查文件是否存在，避免 loadmat 抛出不直观错误。
	if not MAT_PATH.exists():
		raise FileNotFoundError(f"MAT file not found: {MAT_PATH}")

	mat = loadmat(MAT_PATH)
	# 过滤掉以双下划线开头的内部元信息键，只显示用户存储的变量名。
	keys = [k for k in mat.keys() if not k.startswith("__")]

	print(f"文件路径: {MAT_PATH}")
	print(f"变量键: {keys}")

	if "label" not in mat:
		raise KeyError(f"{MAT_PATH} does not contain key 'label'")

	raw_label = np.asarray(mat["label"])
	# 拉平成一维，无论原始是行向量、列向量还是多维，都变成一维序列，方便后续统计与导出。
	labels = raw_label.reshape(-1)
    # 返回去重后的标签值及每个值出现的次数。
	unique, counts = np.unique(labels, return_counts=True)

	print(f"原始 label 形状: {raw_label.shape}, 数据类型: {raw_label.dtype}")
	print(f"展平后形状: {labels.shape}")
	print(f"标签分布: {dict(zip(unique.tolist(), counts.tolist()))}")

	# 预览前 20 个标签值，快速观察是否存在分段结构。
	preview = max(0, min(20, labels.shape[0]))
	print(f"前 {preview} 个标签: {labels[:preview].tolist()}")

	# 统计标签切换点数量，用于判断是否是“整段标签”排列。
    # 通过比较相邻元素（labels[1:] 与 labels[:-1]）得到布尔数组，np.where 返回不相等的索引位置。
	changes = np.where(labels[1:] != labels[:-1])[0]
	print(f"标签变化点数量: {len(changes)}")
	if len(changes) > 0:
		print(f"第一个变化点索引: {int(changes[0])}")

	# 导出为 CSV，供后续脚本使用。输出字段：index,label。
	OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["index", "label"])
		for idx, label in enumerate(labels):
			writer.writerow([idx, int(label)])
	print(f"已导出 CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
