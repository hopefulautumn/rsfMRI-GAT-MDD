from __future__ import annotations

"""基于已训练 GAT 模型提取关键功能连接边。

思路：
1. 读取与训练一致的图构建配置，重建图数据。
2. 逐 fold 加载对应 checkpoint。
3. 在验证集上提取第一层注意力权重。
4. 将有向边注意力汇总为无向边重要性，跨样本/跨 fold 聚合。
5. 输出 Top-N 关键边到 CSV，供后续脑区解释。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from fc_to_graph_dataset import GraphBuildConfig, build_graph_dataset


class GATClassifier(nn.Module):
    """与训练脚本保持一致的 GAT 分类器结构。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        # 注意：结构参数必须和训练时完全一致，否则无法加载 checkpoint。
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        self.gat2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        self.fc = nn.Linear(hidden_channels * num_heads, 2)

    def forward(self, data):
        """标准前向，用于完整推理（本脚本主要关注注意力提取）。"""

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)

    def get_layer1_attention(self, data):
        """返回第一层注意力 (edge_index, alpha)。

        返回值说明：
        - att_edge_index: [2, E]，注意力对应的边索引。
        - alpha: [E, heads]，每条边在各注意力头上的权重。
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        _, (att_edge_index, alpha) = self.gat1(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        return att_edge_index, alpha


def parse_args() -> argparse.Namespace:
    """命令行参数。"""

    parser = argparse.ArgumentParser(description="Extract stable important edges from trained GAT folds")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("outputs/gat_cv/checkpoints"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/gat_cv/important_edges.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--node-feature-mode", type=str, default="fc_row", choices=["fc_row", "ones"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """主流程：重建图 -> 逐fold提取注意力 -> 聚合排序 -> 导出CSV。"""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GraphBuildConfig(
        threshold=args.threshold,
        top_k=(args.top_k if args.top_k > 0 else None),
        node_feature_mode=args.node_feature_mode,
    )

    graphs, labels = build_graph_dataset(
        args.npz_path,
        config=config,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
    )

    # 输入维度由节点特征模式决定。
    in_channels = graphs[0].x.shape[1]
    # 这里使用与训练一致的分层划分，确保 fold 与 checkpoint 对应。
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # edge_score_sum：累计注意力分数；edge_count：该边被统计次数。
    edge_score_sum: dict[tuple[int, int], float] = defaultdict(float)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)

    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        ckpt_path = args.checkpoints_dir / f"fold_{fold_idx}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = GATClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # 仅在该 fold 的验证子集上提取注意力，避免训练集偏置解释结果。
        val_graphs = [graphs[i] for i in val_idx]
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                att_edge_index, alpha = model.get_layer1_attention(batch)

                # alpha shape: [E, heads]；对多头取平均。
                alpha_mean = alpha.mean(dim=1).detach().cpu().numpy()
                e_idx = att_edge_index.detach().cpu().numpy()

                for edge_i in range(e_idx.shape[1]):
                    u = int(e_idx[0, edge_i])
                    v = int(e_idx[1, edge_i])
                    if u == v:
                        # 跳过自环。
                        continue
                    # 无向边聚合：把 (u,v) 与 (v,u) 视为同一条边。
                    key = (u, v) if u < v else (v, u)
                    edge_score_sum[key] += float(alpha_mean[edge_i])
                    edge_count[key] += 1

    # 计算每条边的平均注意力分数。
    results = []
    for edge, score_sum in edge_score_sum.items():
        cnt = edge_count[edge]
        results.append((edge[0], edge[1], score_sum / cnt, cnt))

    # 按重要性从高到低排序。
    results.sort(key=lambda x: x[2], reverse=True)
    top_n = max(1, args.top_n)
    results = results[:top_n]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # support_count 可反映边在聚合中出现的稳定性。
        writer.writerow(["roi_u", "roi_v", "mean_attention", "support_count"])
        for row in results:
            writer.writerow(row)

    print(f"Saved important edges: {args.output_csv}")
    print(f"Total exported edges: {len(results)}")


if __name__ == "__main__":
    main()
