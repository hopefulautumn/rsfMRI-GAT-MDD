from __future__ import annotations

"""GAT 图分类训练脚本（7:1:2 分层划分版本）。

1. 按 7:1:2 分层划分 train/val/test。
2. 训练集用于参数更新，验证集用于早停与模型选择，测试集仅用于最终评估。
3. 使用 Adam + CrossEntropyLoss + weight decay。
4. 使用学习率衰减（每 50 epoch 衰减到 0.5 倍）。
5. 使用早停（监控验证集 loss）。
6. 输出 Accuracy / Precision / Recall / F1 / AUC。

"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from fc_to_graph_dataset import GraphBuildConfig, build_graph_dataset


class GATClassifier(nn.Module):
    """用于图级二分类的两层 GATv2 模型。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # dropout 会在每层图卷积后的特征上生效，用于缓解过拟合。
        self.dropout = dropout

        # 第1层图注意力卷积：
        # 输入维度: in_channels
        # 输出维度: hidden_channels * num_heads（因为 concat=True）
        # edge_dim=1 表示每条边附带 1 维边特征（FC 权重）。
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        # 第2层图注意力卷积：继续在图结构上聚合高阶邻域信息。
        # 输入维度要和第1层输出一致，因此是 hidden_channels * num_heads。
        self.gat2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        # 图级分类头：将池化后的图表示映射到 2 类 logits（MDD/HC）。
        self.fc = nn.Linear(hidden_channels * num_heads, 2)

    def forward(self, data):
        """前向传播。

        data 是 PyG 的批量图对象，包含：
        - x: 节点特征，形状 [所有图节点总数, F]
        - edge_index: 边索引，形状 [2, E]
        - edge_attr: 边特征，形状 [E, 1]
        - batch: 每个节点属于哪张图的索引，形状 [所有图节点总数]
        """

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # 第一层图注意力聚合（显式使用 edge_attr 作为边权特征）。
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        # ELU 常用于 GAT/GCN，能提供稳定非线性。
        x = F.elu(x)
        # 仅训练阶段启用 dropout；推理阶段自动关闭。
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层图注意力聚合。
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级任务关键步骤：把“节点级表示”聚合成“图级表示”。
        # 这里用 global_mean_pool，即同一图内节点取均值。
        x = global_mean_pool(x, batch)
        # 输出 logits（未过 softmax，供 CrossEntropyLoss 使用）。
        return self.fc(x)


def set_seed(seed: int) -> None:
    """统一设置随机种子，尽量提高实验可复现性。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_node_normalizer(graphs) -> tuple[torch.Tensor, torch.Tensor]:
    """仅用训练集拟合节点特征标准化参数（零均值）。"""

    # 把训练集中所有图的节点特征拼接在一起，按“特征维度”统计均值和标准差。
    x_all = torch.cat([g.x for g in graphs], dim=0)
    mean = x_all.mean(dim=0, keepdim=True)
    std = x_all.std(dim=0, keepdim=True)
    # 某些特征可能方差为 0，防止除零。
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


def apply_node_normalizer(graphs, mean: torch.Tensor, std: torch.Tensor):
    """将训练集拟合得到的均值/方差应用到任意子集。"""

    normalized = []
    for g in graphs:
        # clone 避免原图对象被原地修改，便于复用原始数据。
        g2 = g.clone()
        g2.x = (g2.x - mean) / std
        normalized.append(g2)
    return normalized


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    """在指定数据集上评估模型并返回指标。"""

    # 评估模式：关闭 dropout/bn 的训练行为。
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        # 评估阶段不做反向传播，减少显存与计算开销。
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            # 图分类任务里每张图一个标签，因此用 batch.y.view(-1) 对齐 logits 维度。
            loss = criterion(logits, batch.y.view(-1))
            total_loss += float(loss.item())

            # 第 1 类概率（这里约定为正类）的分数用于计算 ROC-AUC。
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(batch.y.view(-1).cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    # zero_division=0：当某些极端批次没有预测到正类时，避免报错并返回 0。
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        # 只有当标签中同时包含两类时，AUC 才有定义。
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "loss": avg_loss,
    }
    return avg_loss, metrics


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。

    参数分组说明：
    - 数据与输出路径：npz_path / output_dir
    - 划分策略：train_ratio / val_ratio / test_ratio
    - 模型与训练：epochs / batch_size / lr / weight_decay / hidden_channels / num_heads / dropout
    - 学习率与早停：lr_step_size / lr_gamma / early_stop_*
    - 构图配置：threshold / top_k / node_feature_mode
    - 复现与联调：seed / max_samples
    """

    parser = argparse.ArgumentParser(description="Train GAT with stratified 7:1:2 split")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gat_split_712"))

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--node-feature-mode", type=str, default="fc_row", choices=["fc_row", "ones"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """主流程。

    执行顺序：
    1) 解析参数并设种子
    2) 构图
    3) 分层 7:1:2 划分
    4) 训练集拟合标准化并应用到 val/test
    5) 训练 + 验证（含学习率衰减和早停）
    6) 加载最佳模型并在测试集评估
    7) 保存结果 JSON 和 checkpoint
    """

    args = parse_args()
    set_seed(args.seed)

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

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

    # 索引数组用于做 train/val/test 划分，避免复制大对象。
    labels = np.asarray(labels)
    indices = np.arange(len(labels))

    # 第一步：先划分测试集（20%）
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    # 第二步：在剩余 80% 内再划分训练/验证（使总体达到 7:1）
    val_ratio_within_train_val = args.val_ratio / (args.train_ratio + args.val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        random_state=args.seed,
        stratify=labels[train_val_idx],
    )

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # 仅用训练集拟合零均值标准化参数，避免数据泄漏。
    mean, std = fit_node_normalizer(train_graphs)
    train_graphs = apply_node_normalizer(train_graphs, mean, std)
    val_graphs = apply_node_normalizer(val_graphs, mean, std)
    test_graphs = apply_node_normalizer(test_graphs, mean, std)

    in_channels = train_graphs[0].x.shape[1]
    print(f"样本总数: {len(graphs)} | 输入维度: {in_channels}")
    print(f"划分结果: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    output_dir = args.output_dir
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 单次 7:1:2 实验只保存一个最优模型。
    best_ckpt = ckpt_dir / "best_model.pt"

    model = GATClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # StepLR：每 step_size 个 epoch，把学习率乘以 gamma（如 50 epoch * 0.5）。
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_val_metrics = None
    best_epoch = 0
    # 连续多少个 epoch 没有提升（用于早停计数）。
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            # 反向传播 + 参数更新。
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        # ---- 验证阶段 ----
        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        # 每 5 个 epoch 打印一次，同时也打印首末 epoch，便于观察训练趋势。
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['auc']:.4f}"
            )

        # 早停依据：验证集 loss 是否显著下降
        if val_loss < (best_val_loss - args.early_stop_min_delta):
            # 验证损失有显著改进：刷新最优记录并保存模型。
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            # 没有改进：计数 +1，等待触发早停。
            no_improve_epochs += 1

        # 每个 epoch 结束后更新学习率。
        scheduler.step()

        if no_improve_epochs >= args.early_stop_patience:
            print(f"触发早停：验证集 loss 连续 {no_improve_epochs} 个 epoch 未改善。")
            break

    # 使用验证集最优模型在测试集上做一次最终评估。
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    _, test_metrics = evaluate(model, test_loader, device)

    serializable_config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    result = {
        # 记录数据划分规模和比例，保证实验可复现。
        "split": {
            "train_size": len(train_graphs),
            "val_size": len(val_graphs),
            "test_size": len(test_graphs),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        # 这里是最终汇报用的测试集指标。
        "test_metrics": test_metrics,
        # 保存完整配置，便于后续追溯实验条件。
        "config": serializable_config,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "split_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 7:1:2 测试集结果 =====")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1       : {test_metrics['f1']:.4f}")
    print(f"AUC      : {test_metrics['auc']:.4f}")
    print(f"已保存指标: {metrics_path}")
    print(f"已保存最优模型: {best_ckpt}")


if __name__ == "__main__":
    main()
