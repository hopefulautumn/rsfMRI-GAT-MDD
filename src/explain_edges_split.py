from __future__ import annotations

"""基于 7:1:2 训练结果提取关键功能连接边。

这个脚本属于“模型解释层”，核心任务不是重新训练模型，
而是把已经训练好的 7:1:2 GAT 模型中学到的注意力权重提取出来，
从而找出“哪些 ROI 连接最重要”。

与 train_gat_split.py 对齐的关键点：
1. 使用相同的分层 7:1:2 划分方式（相同 seed）。
2. 仅用训练集拟合节点特征标准化参数，再应用到 val/test。
3. 加载单个 best_model.pt，而不是按 fold 循环。
4. 在指定子集（train/val/test）上提取第一层注意力并聚合输出。

模型到底关注了哪些脑连接。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from fc_to_graph_dataset import GraphBuildConfig, build_graph_dataset


class GATClassifier(nn.Module):
    """与 7:1:2 训练脚本保持一致的图级二分类模型。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # dropout 保存为成员变量，forward 中会多次使用。
        self.dropout = dropout

        # 第 1 层 GATv2：负责把原始节点特征与邻居信息进行第一次融合。
        # edge_dim=1 表示边上携带 1 维特征，也就是 FC 值（经过构图阶段保留下来的边权）。
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )

        # 第 2 层 GATv2：进一步在更高阶的图结构上聚合信息。
        # 第一层输出维度是 hidden_channels * num_heads，因此第二层输入必须匹配这个维度。
        self.gat2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )

        # 图级分类头：把池化后的整张图表示映射到 2 个 logits。
        # 2 类分别对应二分类任务中的两个标签（MDD / HC）。
        self.fc = nn.Linear(hidden_channels * num_heads, 2)

    def forward(self, data):
        """标准前向传播。

        参数
        ----
        data:
            PyG 的批量图对象，通常包含：
            - x: 节点特征
            - edge_index: 边索引
            - edge_attr: 边特征/边权
            - batch: 每个节点属于哪张图

        返回
        ----
        logits:
            图级二分类的原始输出分数，供 softmax / CrossEntropyLoss 使用。
        """

        # batch 记录“哪些节点属于同一张图”。
        # 因为 DataLoader 会把多张图拼成一个 batch，所以必须靠 batch 来恢复图的归属关系。
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一次图卷积：用邻居和边权更新节点表示。
        # 这里不仅利用图拓扑，还利用 FC 值作为边特征。
        x = self.gat1(x, edge_index, edge_attr=edge_attr)

        # 激活函数 ELU 让特征表达更非线性，同时比简单 ReLU 更平滑。
        x = F.elu(x)

        # dropout 仅在训练阶段生效，推理/解释阶段会自动关闭。
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二次图卷积：继续聚合信息，形成更抽象的节点表示。
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化：把一张图里的所有节点表示汇总成一个图向量。
        # global_mean_pool 的直觉就是“同一张图里的节点取平均”。
        x = global_mean_pool(x, batch)

        # 最后输出 logits，而不是概率。
        # 这样做是为了直接和 CrossEntropyLoss 对接，避免重复做 softmax。
        return self.fc(x)

    def get_layer1_attention(self, data):
        """提取第一层注意力权重。

        返回
        ----
        att_edge_index:
            注意力权重对应的边索引，形状 [2, E]。
        alpha:
            每条边在每个注意力头上的权重，形状 [E, heads]。

        意义
        ----
        这个函数是解释脚本的核心之一：
        它让我们能看到模型第一层到底对哪些边“更关注”。
        """

        # 这里不需要 batch 信息，因为我们只关心注意力和边的对应关系。
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # return_attention_weights=True 会让 GATv2Conv 同时返回注意力信息。
        # 返回结果中包含“边索引 + 注意力系数”。
        _, (att_edge_index, alpha) = self.gat1(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        return att_edge_index, alpha


def fit_node_normalizer(graphs) -> tuple[torch.Tensor, torch.Tensor]:
    """仅用训练集拟合节点特征标准化参数。

    为什么要做这一步？
    - 如果节点特征没有标准化，不同维度的数值尺度可能差异很大，训练不稳定。
    - 只用训练集拟合 mean/std 是为了避免数据泄漏。
      也就是说，验证集/测试集不能参与“标准化参数”的统计。

    返回
    ----
    mean, std:
        训练集节点特征的均值和标准差，后续会用于所有子集。
    """

    # 把所有训练图的节点特征拼到一起，统一按特征维度统计均值和标准差。
    x_all = torch.cat([g.x for g in graphs], dim=0)
    mean = x_all.mean(dim=0, keepdim=True)
    std = x_all.std(dim=0, keepdim=True)

    # 某些特征维度可能方差为 0（例如全常数），这里把 0 替换成 1，避免除零。
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


def apply_node_normalizer(graphs, mean: torch.Tensor, std: torch.Tensor):
    """对给定图列表应用训练集标准化参数。

    原则是：
    - 训练集负责“学习”标准化参数；
    - 验证集和测试集只“使用”这些参数，不再自己计算 mean/std。

    这样才能保证评估是公平的，不把测试信息泄漏回训练过程。
    """

    normalized = []
    for g in graphs:
        # clone() 的目的是不直接修改原始图对象，避免副作用。
        g2 = g.clone()
        # 标准化公式：(x - mean) / std
        # 这样每一维特征都被拉到统一尺度上，更利于模型训练。
        g2.x = (g2.x - mean) / std
        normalized.append(g2)
    return normalized


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    有两个层面的参数：
    1. 与训练对齐的参数（seed、split 比例、top_k、node_feature_mode 等）；
    2. 与解释相关的参数（checkpoint、输出路径、top_n、split_target 等）。

    这样设计的目的是：只要训练配置没变，解释脚本就能完全复现训练时的数据划分。
    """

    parser = argparse.ArgumentParser(description="Extract important edges from 7:1:2 trained GAT model")

    # 输入的 FC 数据包，必须和训练时使用的是同一份。
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))

    # 训练好的模型权重文件（通常是 best_model.pt）。
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("outputs/gat_split_712_formal/checkpoints/best_model.pt"),
    )

    # 导出的边级重要性结果。
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/gat_split_712_formal/important_edges_test.csv"),
    )

    # 指定解释哪一部分数据：训练集、验证集还是测试集。
    # 默认是 test，因为通常最关心模型泛化到未见数据时的解释结果。
    parser.add_argument("--split-target", type=str, default="test", choices=["train", "val", "test"])

    # 这三个比例必须与训练脚本保持一致，否则解释时会切出不同的样本集合。
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    # 解释阶段的 batch_size。
    # 默认设为 1，是为了让注意力提取更直观，也方便逐图处理。
    parser.add_argument("--batch-size", type=int, default=1)

    # 这些结构参数必须和训练时完全一致，否则加载 checkpoint 会失败。
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # 构图策略参数：与训练脚本一致。
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--node-feature-mode", type=str, default="fc_row", choices=["fc_row", "ones"])

    # 随机种子必须一致，才能保证 split 结果与训练时完全对齐。
    parser.add_argument("--seed", type=int, default=42)

    # smoke / 调试时可限制样本数；正式解释时通常保持 0，即使用全部样本。
    parser.add_argument("--max-samples", type=int, default=0)

    # 输出多少条最重要的边。
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """主流程：

    1. 解析参数
    2. 重建和训练一致的数据划分
    3. 用训练集拟合标准化参数，并应用到各子集
    4. 加载训练好的 best_model.pt
    5. 在指定子集上提取第一层注意力
    6. 把有向边汇总为无向边重要性
    7. 导出 CSV
    """

    args = parse_args()

    # 基本参数一致性检查：训练/验证/测试比例必须加起来等于 1。
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    # 自动选择设备：有 GPU 用 GPU，否则用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 构图参数必须和训练时保持一致。
    config = GraphBuildConfig(
        threshold=args.threshold,
        top_k=(args.top_k if args.top_k > 0 else None),
        node_feature_mode=args.node_feature_mode,
    )

    # 从 npz 文件中重建图数据集。
    # graphs: 图列表；labels: 每张图对应的标签。
    graphs, labels = build_graph_dataset(
        args.npz_path,
        config=config,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
    )

    # 转为 numpy，便于做分层抽样。
    labels = np.asarray(labels)
    # 这里的 indices 只是样本索引，不是脑区索引。
    indices = np.arange(len(labels))

    # 与 train_gat_split.py 对齐：先拆 test，再在 train_val 中拆 val。
    # 这是实现 7:1:2 的关键步骤。
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    # 在剩余样本中继续拆分出验证集。
    # 之所以要先拿出 test，再从 train_val 中拆 val，是为了最终比例严格对齐 7:1:2。
    val_ratio_within_train_val = args.val_ratio / (args.train_ratio + args.val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        random_state=args.seed,
        stratify=labels[train_val_idx],
    )

    # 按索引切出三部分图数据。
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # 与训练对齐：仅用训练集拟合标准化参数。
    # 这一步非常重要，因为如果把验证集/测试集也拿来统计 mean/std，
    # 就等于把未来信息偷偷泄漏给模型了。
    mean, std = fit_node_normalizer(train_graphs)
    train_graphs = apply_node_normalizer(train_graphs, mean, std)
    val_graphs = apply_node_normalizer(val_graphs, mean, std)
    test_graphs = apply_node_normalizer(test_graphs, mean, std)

    # 根据 split_target 决定到底解释哪一部分数据。
    if args.split_target == "train":
        target_graphs = train_graphs
    elif args.split_target == "val":
        target_graphs = val_graphs
    else:
        target_graphs = test_graphs

    # 输入维度由节点特征模式决定。
    # 如果 node_feature_mode 是 fc_row，则维度通常等于 ROI 数；
    # 如果是 ones，则维度为 1。
    in_channels = target_graphs[0].x.shape[1]

    # 打印划分情况，便于确认 split 是否与训练阶段一致。
    print(
        f"样本总数: {len(graphs)} | 划分: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}"
    )
    print(f"解释子集: {args.split_target} | 样本数: {len(target_graphs)}")

    # checkpoint 必须存在，否则无法载入训练好的模型。
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {args.checkpoint_path}")

    # 重新构建一个与训练时完全一致的模型结构。
    model = GATClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    # 载入训练好的权重参数。
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    # 切换到 eval 模式，关闭 dropout 等训练专属行为。
    model.eval()

    # 逐图处理，方便聚合每个样本中的注意力边。
    loader = DataLoader(target_graphs, batch_size=args.batch_size, shuffle=False)

    # edge_score_sum：无向边 (u, v) 的注意力分数累加。
    # edge_count：该无向边在多少个样本/批次中出现过，用于计算 support_count。
    edge_score_sum: dict[tuple[int, int], float] = defaultdict(float)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)

    with torch.no_grad():
        # 解释阶段只做前向传播，不需要梯度。
        for batch in loader:
            batch = batch.to(device)
            # 提取第一层注意力权重。
            att_edge_index, alpha = model.get_layer1_attention(batch)

            # 对多头注意力取平均，得到每条边的综合注意力分数。
            alpha_mean = alpha.mean(dim=1).detach().cpu().numpy()
            # 转回 CPU 方便用 numpy 遍历和写 CSV。
            e_idx = att_edge_index.detach().cpu().numpy()

            for edge_i in range(e_idx.shape[1]):
                # 单条有向边的两个端点。
                u = int(e_idx[0, edge_i])
                v = int(e_idx[1, edge_i])
                # 跳过自环：自连接通常不作为有意义的脑区连接解释。
                if u == v:
                    continue

                # 将有向边折叠成无向边：
                # (u, v) 和 (v, u) 统一记作同一条边。
                key = (u, v) if u < v else (v, u)

                # 累加注意力分数。
                edge_score_sum[key] += float(alpha_mean[edge_i])
                # 统计这条边被看见了多少次，用于 support_count。
                edge_count[key] += 1

    # 把累加结果整理成最终输出格式。
    results = []
    for edge, score_sum in edge_score_sum.items():
        # support_count：这条边在聚合中出现的次数。
        cnt = edge_count[edge]
        # mean_attention：平均注意力分数。
        results.append((edge[0], edge[1], score_sum / cnt, cnt))

    # 按平均注意力从高到低排序。
    results.sort(key=lambda x: x[2], reverse=True)

    # 只保留前 top-n 条重要边。
    results = results[: max(1, args.top_n)]

    # 输出目录不存在时自动创建。
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 输出列含义：
        # roi_u / roi_v       -> 边的两个端点编号
        # mean_attention      -> 平均注意力
        # support_count       -> 聚合中出现次数
        writer.writerow(["roi_u", "roi_v", "mean_attention", "support_count"])
        for row in results:
            writer.writerow(row)

    # 终端提示：告诉用户关键边文件已经成功生成。
    print(f"已保存关键边: {args.output_csv}")
    print(f"导出边数量: {len(results)}")


if __name__ == "__main__":
    main()
