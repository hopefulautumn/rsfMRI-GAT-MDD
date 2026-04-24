from __future__ import annotations

"""GAT 图分类训练脚本（LOSO: Leave-One-Site-Out）。

每次留出一个站点作为外部测试集，其余站点用于训练。
为避免使用测试站点调参，每轮 LOSO 在训练站点中再划分一个内部验证集，
按内部验证 AUC 保存最佳模型，再在外部测试站点评估。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from fc_to_graph_dataset import GraphBuildConfig, build_graph_dataset, load_fc_npz


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
        self.dropout = dropout
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
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
            loss = criterion(logits, batch.y.view(-1))
            total_loss += float(loss.item())

            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(batch.y.view(-1).cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    has_both_classes = len(set(all_labels)) > 1
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": float(roc_auc_score(all_labels, all_probs)) if has_both_classes else float("nan"),
        "loss": float(avg_loss),
        "n_samples": int(len(all_labels)),
    }
    return avg_loss, metrics


def split_inner_train_val(
    train_indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
    inner_val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在训练站点内做分层划分，用于模型选择。"""

    train_labels = labels[train_indices]
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=seed)
        inner_train_local, inner_val_local = next(sss.split(np.zeros(len(train_indices)), train_labels))
        return train_indices[inner_train_local], train_indices[inner_val_local]
    except ValueError:
        # 极端情况下（某类样本太少）回退为简单切分，避免流程中断。
        n = len(train_indices)
        n_val = max(1, int(round(n * inner_val_ratio)))
        perm = np.random.RandomState(seed).permutation(n)
        val_local = perm[:n_val]
        tr_local = perm[n_val:]
        if len(tr_local) == 0:
            tr_local = perm[:1]
            val_local = perm[1:]
        return train_indices[tr_local], train_indices[val_local]


def train_one_site(
    site_name: str,
    train_graphs,
    inner_val_graphs,
    test_graphs,
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
) -> dict:
    """训练单个 LOSO 轮次并在外部测试站点评估。"""

    model = GATClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(inner_val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    best_auc = -np.inf
    best_epoch = -1
    ckpt_path = checkpoint_dir / f"site_{site_name}.pt"

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
        val_auc = val_metrics["auc"]
        score = val_auc if not np.isnan(val_auc) else -np.inf
        if score >= best_auc:
            best_auc = score
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            train_loss = running_loss / max(len(train_loader), 1)
            val_auc_disp = "nan" if np.isnan(val_auc) else f"{val_auc:.4f}"
            print(
                f"Site {site_name} Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"inner_val_auc={val_auc_disp} | inner_val_f1={val_metrics['f1']:.4f}"
            )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_metrics = evaluate(model, test_loader, device)
    test_metrics["site"] = site_name
    test_metrics["best_epoch"] = int(best_epoch)
    return test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAT with LOSO (Leave-One-Site-Out)")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gat_loso"))
    parser.add_argument("--epochs", type=int, default=50)
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
        help="站点校正方式。LOSO 中 combat 默认不支持（测试站点未见）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--inner-val-ratio", type=float, default=0.1)
    return parser.parse_args()


def compute_summary(site_metrics: list[dict]) -> dict:
    auc_values = np.asarray([m["auc"] for m in site_metrics], dtype=float)
    f1_values = np.asarray([m["f1"] for m in site_metrics], dtype=float)
    acc_values = np.asarray([m["accuracy"] for m in site_metrics], dtype=float)
    n_values = np.asarray([m["n_samples"] for m in site_metrics], dtype=float)

    weights = n_values / n_values.sum()

    macro = {
        "auc": float(np.nanmean(auc_values)),
        "f1": float(np.mean(f1_values)),
        "accuracy": float(np.mean(acc_values)),
    }
    weighted = {
        "auc": float(np.nansum(auc_values * weights)),
        "f1": float(np.sum(f1_values * weights)),
        "accuracy": float(np.sum(acc_values * weights)),
    }
    return {
        "n_sites": int(len(site_metrics)),
        "macro_mean": macro,
        "weighted_mean": weighted,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.site_harmonization == "combat":
        raise ValueError(
            "LOSO 测试站点在训练时不可见，标准 ComBat 无法对未见站点 transform。"
            "建议在 LOSO 中使用 --site-harmonization none。"
        )

    if not (0.0 < args.inner_val_ratio < 0.5):
        raise ValueError("--inner-val-ratio 必须在 (0, 0.5) 区间")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    _, labels_from_npz, _, site_ids = load_fc_npz(args.npz_path)

    if args.max_samples > 0:
        labels_from_npz = labels_from_npz[: args.max_samples]
        site_ids = site_ids[: args.max_samples]

    if len(graphs) == 0:
        raise ValueError("图数据集为空，请检查 npz_path 或 max_samples")

    if len(labels_from_npz) != len(graphs):
        raise ValueError(
            "标签长度不一致：build_graph_dataset 与 load_fc_npz 结果不匹配，"
            f"got {len(graphs)} vs {len(labels_from_npz)}"
        )

    if np.all(site_ids == "UNK"):
        raise ValueError("site_ids 不可用，请先运行 src/build_fc_dataset.py 生成包含 site_ids 的 npz")

    labels = labels_from_npz
    first_x = graphs[0].x
    if first_x is None or first_x.ndim != 2:
        raise ValueError("无效节点特征：graph.x 为空或形状不是二维")

    in_channels = int(first_x.shape[1])
    unique_sites = np.unique(site_ids)
    print(f"Loaded {len(graphs)} graphs | in_channels={in_channels} | n_sites={len(unique_sites)}")

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for site in unique_sites:
        test_mask = site_ids == site
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        inner_train_idx, inner_val_idx = split_inner_train_val(
            train_idx,
            labels,
            seed=args.seed,
            inner_val_ratio=args.inner_val_ratio,
        )

        train_graphs = [graphs[i] for i in inner_train_idx]
        inner_val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        print(
            f"\n===== LOSO Site {site} =====\n"
            f"train={len(train_graphs)} | inner_val={len(inner_val_graphs)} | test={len(test_graphs)}"
        )

        site_metrics = train_one_site(
            site_name=str(site),
            train_graphs=train_graphs,
            inner_val_graphs=inner_val_graphs,
            test_graphs=test_graphs,
            in_channels=in_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_channels=args.hidden_channels,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        all_metrics.append(site_metrics)

        auc_disp = "nan" if np.isnan(site_metrics["auc"]) else f"{site_metrics['auc']:.4f}"
        print(
            f"Site {site} test | auc={auc_disp} | f1={site_metrics['f1']:.4f} | "
            f"acc={site_metrics['accuracy']:.4f}"
        )

    summary = compute_summary(all_metrics)
    serializable_config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    summary_payload = {
        "summary": summary,
        "site_metrics": all_metrics,
        "config": serializable_config,
    }

    with (output_dir / "loso_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "loso_site_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["site", "n_samples", "accuracy", "precision", "recall", "f1", "auc", "loss", "best_epoch"],
        )
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)

    print("\n===== LOSO Summary =====")
    print(f"Macro AUC: {summary['macro_mean']['auc']:.4f}")
    print(f"Macro F1 : {summary['macro_mean']['f1']:.4f}")
    print(f"Macro Acc: {summary['macro_mean']['accuracy']:.4f}")
    print(f"Weighted AUC: {summary['weighted_mean']['auc']:.4f}")
    print(f"Weighted F1 : {summary['weighted_mean']['f1']:.4f}")
    print(f"Weighted Acc: {summary['weighted_mean']['accuracy']:.4f}")
    print(f"Saved: {output_dir / 'loso_summary.json'}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
