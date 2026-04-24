from __future__ import annotations

"""GAT 图分类训练脚本（分层交叉验证版本）。

设计目标：
1. 避免交叉验证信息泄漏：每个 fold 都重新初始化模型和优化器。
2. 支持带边权的 GATv2（edge_dim=1，对应 FC 边权）。
3. 输出可复用产物：
    - 每折最佳 checkpoint（用于后续解释关键连接）。
    - 全部折的指标汇总 JSON（用于论文表格和统计）。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from fc_to_graph_dataset import GraphBuildConfig, fc_to_graph, load_fc_npz
from site_harmonization import CombatHarmonizer


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
        # dropout 概率在两层注意力后都使用。
        self.dropout = dropout
        # 第1层：输入维度 in_channels -> hidden_channels * num_heads。
        # edge_dim=1 表示每条边有 1 维边特征（FC 权重）。
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        # 第2层：继续在图结构上聚合信息。
        self.gat2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        # 图级分类头，输出2类 logits。
        self.fc = nn.Linear(hidden_channels * num_heads, 2)

    def forward(self, data):
        """前向传播：节点更新 -> 图池化 -> 分类。"""

        # batch 向量由 PyG DataLoader 提供，标记每个节点属于哪张图。
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # 注意力卷积会同时利用拓扑和 edge_attr（FC 边权）。
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 图级任务常用全局池化：把节点表示汇总成图表示。
        x = global_mean_pool(x, batch)
        return self.fc(x)


def set_seed(seed: int) -> None:
    """固定随机种子，提升结果复现性。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    """在验证集上评估模型并返回常见二分类指标。"""

    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            # batch.y 在图分类任务中是每张图一个标签。
            loss = criterion(logits, batch.y.view(-1))
            total_loss += float(loss.item())

            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(batch.y.view(-1).cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "loss": avg_loss,
    }
    return avg_loss, metrics


def train_one_fold(
    fold_id: int,
    train_graphs,
    val_graphs,
    in_channels: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_channels: int,
    num_heads: int,
    dropout: float,
    device: torch.device,
    checkpoint_dir: Path,
):
    """训练单个 fold，并保存该 fold 的最佳 checkpoint（按 AUC）。"""

    # 关键点：每个 fold 都新建模型，避免跨 fold 泄漏。
    model = GATClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    best_auc = -1.0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        _, val_metrics = evaluate(model, val_loader, device)
        # 按验证 AUC 选择最佳模型并持久化。
        if val_metrics["auc"] >= best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics
            ckpt_path = checkpoint_dir / f"fold_{fold_id}.pt"
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            train_loss = running_loss / max(len(train_loader), 1)
            print(
                f"Fold {fold_id} Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_f1={val_metrics['f1']:.4f}"
            )

    return best_metrics


def parse_args() -> argparse.Namespace:
    """命令行参数定义。"""

    parser = argparse.ArgumentParser(description="Train GAT with leakage-safe stratified CV")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gat_cv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--node-feature-mode", type=str, default="fc_row", choices=["fc_row", "ones"])
    parser.add_argument(
        "--site-harmonization",
        type=str,
        default="none",
        choices=["none", "combat"],
        help="站点校正方式。combat 仅在验证集站点被训练集覆盖时可用。",
    )
    parser.add_argument(
        "--cv-mode",
        type=str,
        default="auto",
        choices=["auto", "grouped", "stratified"],
        help="CV strategy: auto=prefer grouped if site_ids available, grouped=force StratifiedGroupKFold, stratified=force StratifiedKFold",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def build_graphs_from_fc(fc_matrices: np.ndarray, labels: np.ndarray, config: GraphBuildConfig):
    """把 [N, R, R] FC 批量转换为图对象列表。"""

    return [
        fc_to_graph(fc_matrix, int(label), config)
        for fc_matrix, label in zip(fc_matrices, labels)
    ]


def main() -> None:
    """主流程：构图 -> 分层CV训练 -> 汇总保存。"""

    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = GraphBuildConfig(
        threshold=args.threshold,
        top_k=(args.top_k if args.top_k > 0 else None),
        node_feature_mode=args.node_feature_mode,
    )

    fc_matrices, labels_from_npz, _, site_ids = load_fc_npz(args.npz_path)
    if args.max_samples > 0:
        fc_matrices = fc_matrices[: args.max_samples]
        labels_from_npz = labels_from_npz[: args.max_samples]
        site_ids = site_ids[: args.max_samples]

    labels = labels_from_npz

    if len(fc_matrices) == 0:
        raise ValueError("图数据集为空，请检查 npz_path 或 max_samples 设置")

    # 先用首样本做一次构图，获取输入维度。
    first_graph = fc_to_graph(fc_matrices[0], int(labels[0]), config)
    first_x = first_graph.x
    if first_x is None or first_x.ndim != 2:
        raise ValueError("无效节点特征：graph.x 为空或形状不是二维")

    # 节点输入维度取决于 node_feature_mode。
    in_channels = int(first_x.shape[1])
    print(f"Loaded {len(fc_matrices)} samples | in_channels={in_channels}")

    output_dir = args.output_dir
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 支持可控对照：grouped / stratified / auto。
    unique_sites = np.unique(site_ids)
    has_valid_site_ids = len(unique_sites) > 1 and not np.all(site_ids == "UNK")

    if args.cv_mode == "grouped" and not has_valid_site_ids:
        raise ValueError(
            "--cv-mode grouped 需要有效的 site_ids（且至少两个站点），"
            "请先重新运行 src/build_fc_dataset.py 生成包含 site_ids 的 npz。"
        )

    use_grouped_cv = (args.cv_mode == "grouped") or (
        args.cv_mode == "auto" and has_valid_site_ids
    )

    if args.site_harmonization == "combat" and use_grouped_cv:
        raise ValueError(
            "当前是站点分组评估（grouped/auto->grouped），验证集包含未见站点，"
            "标准 ComBat 无法对未见站点 transform。"
            "如需使用 ComBat，请改用 --cv-mode stratified。"
        )

    if use_grouped_cv:
        splitter = StratifiedGroupKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed,
        )
        split_iter = splitter.split(np.zeros(len(labels)), labels, groups=site_ids)
        print(
            f"Using StratifiedGroupKFold | cv_mode={args.cv_mode} | "
            f"n_splits={args.n_splits} | n_sites={len(unique_sites)}"
        )
    else:
        # stratified 或 auto 但站点信息不可用时，使用普通分层 K 折。
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.zeros(len(labels)), labels)
        reason = (
            "forced by --cv-mode stratified"
            if args.cv_mode == "stratified"
            else "site_ids unavailable or single-site"
        )
        print(f"Using StratifiedKFold | cv_mode={args.cv_mode} | reason={reason}")

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_fc = fc_matrices[train_idx]
        val_fc = fc_matrices[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_sites = site_ids[train_idx]
        val_sites = site_ids[val_idx]

        if args.site_harmonization == "combat":
            harmonizer = CombatHarmonizer()
            train_fc = harmonizer.fit_transform_train(train_fc, train_sites)
            val_fc = harmonizer.transform(val_fc, val_sites)

        train_graphs = build_graphs_from_fc(train_fc, train_labels, config)
        val_graphs = build_graphs_from_fc(val_fc, val_labels, config)

        print(f"\n===== Fold {fold_idx}/{args.n_splits} =====")
        metrics = train_one_fold(
            fold_id=fold_idx,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            in_channels=in_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_channels=args.hidden_channels,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=device,
            checkpoint_dir=ckpt_dir,
        )
        fold_metrics.append(metrics)

    serializable_config = {
        # Path 不能直接 JSON 序列化，这里转字符串。
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    summary = {
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "mean_auc": float(np.mean([m["auc"] for m in fold_metrics])),
        "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
        "std_f1": float(np.std([m["f1"] for m in fold_metrics])),
        "std_auc": float(np.std([m["auc"] for m in fold_metrics])),
        "cv_splitter": splitter.__class__.__name__,
        "n_sites": int(len(unique_sites)),
        "fold_metrics": fold_metrics,
        "config": serializable_config,
    }

    with (output_dir / "cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== CV Summary =====")
    print(f"AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"F1 : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"Acc: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Saved metrics: {output_dir / 'cv_metrics.json'}")


if __name__ == "__main__":
    main()
