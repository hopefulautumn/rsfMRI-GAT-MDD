#!/usr/bin/env python3
"""从已保存的 GAT 模型快速评估站点均衡性。

这个脚本从已训练的检查点加载模型，对测试集进行推理，
然后计算逐样本和逐站点的性能指标。

用法：
    python scripts/eval_site_balance.py \
        --checkpoint outputs/gat_cv_combat_v1/checkpoints/fold_1.pt \
        --npz-path processed/rest_meta_mdd_fc_combat.npz \
        --output-report outputs/gat_cv_combat_v1/fold_1_site_balance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# 需要从你的项目导入
from src.fc_to_graph_dataset import GraphBuildConfig, build_graph_dataset, load_fc_npz
from src.train_gat_cv import GATClassifier
from src.site_balance_validator import (
    compute_site_balanced_metrics,
    print_site_balance_report,
    save_site_balance_report,
)


def evaluate_checkpoint_on_split(
    checkpoint_path: Path,
    graph_dataset,
    site_ids: np.ndarray,
    batch_size: int = 16,
) -> dict:
    """用检查点模型对指定数据集进行推理，返回预测结果。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从检查点的配置恢复模型
    model = GATClassifier(
        in_channels=graph_dataset[0].x.shape[1],
        hidden_channels=64,
        num_heads=4,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(batch.y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    return {
        "labels": np.array(all_labels),
        "predictions": np.array(all_preds),
        "probabilities": np.array(all_probs),
    }


def main():
    parser = argparse.ArgumentParser(description="评估单个 fold 的站点均衡性")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点路径")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc_combat.npz"))
    parser.add_argument("--output-report", type=Path, help="输出报告路径")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--node-feature-mode", type=str, default="fc_row", choices=["fc_row", "ones"])
    args = parser.parse_args()
    
    # 加载数据
    print(f"[INFO] 加载图数据集: {args.npz_path}")
    config = GraphBuildConfig(
        threshold=args.threshold,
        top_k=(args.top_k if args.top_k > 0 else None),
        node_feature_mode=args.node_feature_mode,
    )
    graphs, _ = build_graph_dataset(args.npz_path, config=config)
    _, labels, _, site_ids = load_fc_npz(args.npz_path)
    
    print(f"[INFO] 推理模型: {args.checkpoint}")
    results = evaluate_checkpoint_on_split(
        args.checkpoint,
        graphs,
        site_ids,
        batch_size=args.batch_size,
    )
    
    # 计算站点均衡性
    print(f"[INFO] 计算站点均衡性指标...")
    report = compute_site_balanced_metrics(
        results["labels"],
        results["predictions"],
        results["probabilities"],
        site_ids,
    )
    
    # 打印报告
    print_site_balance_report(report)
    
    # 保存报告
    if args.output_report:
        save_site_balance_report(report, args.output_report)


if __name__ == "__main__":
    main()
