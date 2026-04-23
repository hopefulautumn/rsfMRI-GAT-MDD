from __future__ import annotations

"""将 FC 矩阵数据集转换为 PyTorch Geometric 图数据。

输入：来自预处理阶段的 FC 数据包（npz）。
输出：可被 GAT/GCN 等图模型直接消费的 Data 列表。

映射关系：
每个被试 -> 一张图。
ROI -> 图节点。
ROI-ROI 功能连接强度（FC） -> 图边及边权。

"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphBuildConfig:
    """图构建配置。

    Attributes:
        threshold:
            当未启用 top_k 时，按 |FC| > threshold 选边。
        top_k:
            若为正整数，则每个节点保留 strongest top-k 邻居。
            当 top_k 生效时，threshold 会被忽略。
        node_feature_mode:
            - "fc_row": 使用 FC 的每一行作为节点特征（维度 n_rois）。
            - "ones": 使用全 1 特征（维度 1）。
    """

    # 边阈值（绝对值），可能需要调高来减少稠密图。
    threshold: float = 0.0
    # 每个节点保留的邻居数；None 或 <=0 表示不使用 top-k。
    top_k: int | None = None
    # 节点特征构造模式。
    node_feature_mode: str = "fc_row"  # one of: fc_row, ones


def load_fc_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取预处理输出的 npz 数据包。

    Returns:
        fc_matrices: 形状 [N, R, R]，N 为被试数，R 为 ROI 数。
        labels:      形状 [N]，二分类标签。
        group_names: 形状 [N]，字符串分组名（如 MDD/HC）。
        site_ids:    形状 [N]，站点 ID（如 S01/S20）。
    """

    # allow_pickle=True 用于兼容 object 数组（group_names 可能是 object 类型）。
    data = np.load(npz_path, allow_pickle=True)
    # 统一为 float32，减少显存占用，满足大多数深度学习训练精度需求。
    fc_matrices = np.asarray(data["fc_matrices"], dtype=np.float32)
    # 标签统一为 int64，以兼容 PyTorch 的分类损失函数。
    labels = np.asarray(data["labels"], dtype=np.int64)
    group_names = np.asarray(data["group_names"], dtype=object)
    # 向后兼容：旧版 npz 可能没有 site_ids。
    if "site_ids" in data:
        site_ids = np.asarray(data["site_ids"], dtype=object)
    else:
        site_ids = np.asarray(["UNK"] * len(labels), dtype=object)
    return fc_matrices, labels, group_names, site_ids


def _build_node_features(fc_matrix: np.ndarray, mode: str) -> torch.Tensor:
    """根据模式构建节点特征矩阵 x。

    Args:
        fc_matrix: 单被试 FC 矩阵，形状 [R, R]。
        mode: 节点特征模式。

    Returns:
        x: 形状 [R, F] 的节点特征张量。
    """

    n_rois = fc_matrix.shape[0]
    if mode == "ones":
        # 最简基线：每个节点特征都相同，仅依赖图结构与边权学习。
        return torch.ones((n_rois, 1), dtype=torch.float32)
    if mode == "fc_row":
        # 直接使用每个 ROI 的连接向量作为节点特征。
        # 第 i 行表示 ROI_i 与所有 ROI 的连接模式。
        return torch.tensor(fc_matrix, dtype=torch.float32)
    raise ValueError(f"Unsupported node_feature_mode: {mode}")


def _edge_selector(fc_matrix: np.ndarray, threshold: float, top_k: int | None) -> np.ndarray:
    """根据规则生成边掩码（bool 矩阵）。
    从稠密的 FC 矩阵中选择哪些边（ROI 对）保留到图中。
    因为全连接图（R x R 条边）通常过于稠密，不利于图神经网络的信息传递和计算效率。
    
    规则优先级：
    1. 若 top_k 有效（>0），优先使用 top-k 选边。
    2. 否则使用阈值法 |FC| > threshold。

    Returns:
        mask: 形状 [R, R] 的布尔矩阵，True 表示保留该边。
    """

    n = fc_matrix.shape[0]

    if top_k is not None and top_k > 0:
        # 初始化空掩码。
        mask = np.zeros((n, n), dtype=bool)
        # 以绝对值强度排序（忽略正负方向，仅看连接强弱）。
        abs_fc = np.abs(fc_matrix)
        # 对角线不参与邻居选择（自连接通常不作为功能连接边）。
        np.fill_diagonal(abs_fc, -np.inf)
        for i in range(n):
            # 保留每个节点最强的 top-k 邻居。
            # argpartition 比全排序更高效，适合只取前 k 的场景。
            idx = np.argpartition(abs_fc[i], -top_k)[-top_k:]
            mask[i, idx] = True
        # 无向图：做并集。
        # 若 i 选了 j 或 j 选了 i，都保留 i-j 边。
        mask = np.logical_or(mask, mask.T)
        # 去掉自环。
        np.fill_diagonal(mask, False)
        return mask

    # 阈值法：保留绝对值超过阈值的连接。
    mask = np.abs(fc_matrix) > threshold
    # 去掉自环。
    np.fill_diagonal(mask, False)
    return mask


def fc_to_graph(fc_matrix: np.ndarray, label: int, config: GraphBuildConfig) -> Data:
    """把单个被试的 FC 矩阵转为 PyG 的 Data 对象。"""

    # 选边：得到 [R, R] 的布尔掩码。
    edge_mask = _edge_selector(fc_matrix, threshold=config.threshold, top_k=config.top_k)
    # 把稠密掩码转为 COO 形式的边列表。
    src, dst = np.where(edge_mask)

    # PyG 约定：edge_index 形状 [2, E]。这里直接堆叠 src 和 dst。
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    # 取出对应的 FC 值（即原始相关系数经 Fisher Z 变换后的值）作为边特征，形状 [E, 1]。unsqueeze(1) 使其成为列向量，便于某些卷积层处理
    edge_weight = torch.tensor(fc_matrix[src, dst], dtype=torch.float32).unsqueeze(1)

    # 构建节点特征。
    x = _build_node_features(fc_matrix, config.node_feature_mode)
    # 构建图级标签，形状 [1]，用于图分类任务。
    y = torch.tensor([int(label)], dtype=torch.long)

    # edge_attr 命名沿用 PyG 习惯，供支持边特征的卷积层使用。
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def build_graph_dataset(
    npz_path: Path,
    config: GraphBuildConfig,
    max_samples: int | None = None,
) -> tuple[list[Data], np.ndarray]:
    """批量构建图数据集。

    Args:
        npz_path: FC 数据包路径。
        config: 图构建配置。
        max_samples: 可选；仅取前 max_samples 个样本（用于快速联调）。

    Returns:
        graphs: PyG Data 列表，每个元素对应一个被试。
        labels: 对应标签数组（与 graphs 同顺序）。
    """

    # 读取 FC/标签。
    fc_matrices, labels, _, _ = load_fc_npz(npz_path)

    if max_samples is not None and max_samples > 0:
        # 小样本模式：用于快速 smoke test，不建议用于正式指标。
        fc_matrices = fc_matrices[:max_samples]
        labels = labels[:max_samples]

    graphs: list[Data] = []
    for fc_matrix, label in zip(fc_matrices, labels):
        # 逐样本转换为图对象。
        graphs.append(fc_to_graph(fc_matrix, int(label), config))

    return graphs, labels
