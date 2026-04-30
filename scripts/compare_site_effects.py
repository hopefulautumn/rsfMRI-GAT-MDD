#!/usr/bin/env python3
"""快速验证：对比 ComBat 调和前后的站点效应。

用法：
    python scripts/compare_site_effects.py --baseline outputs/gat_cv_baseline --combat outputs/gat_cv_combat
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from src.site_balance_validator import compute_site_balanced_metrics, print_site_balance_report


def load_cv_results(cv_output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 CV 输出目录加载预测结果和元数据。
    
    假设目录结构为：
        outputs/gat_cv_xxx/
        ├── cv_metrics.json       # 包含 fold_metrics
        └── [fold_idx].predictions.npy (可选)
    """
    
    metrics_file = cv_output_dir / "cv_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"未找到 {metrics_file}")
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    # 从所有 fold 的结果合并出全局预测
    all_labels = []
    all_preds = []
    all_probs = []
    
    for fold_metrics in data.get("fold_metrics", []):
        # 注意：标准 cv_metrics.json 只保存聚合指标，不保存单个预测
        # 这里需要从模型输出重新推理或修改 train_gat_cv.py 来保存预测
        pass
    
    # 临时方案：从源 npz 加载站点信息
    # 在实际使用中，需要修改 train_gat_cv.py 保存逐样本预测
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array([])


def create_comparison_report(
    baseline_dir: Optional[Path],
    combat_dir: Optional[Path],
    site_ids: np.ndarray,
) -> dict:
    """对比两个版本的站点效应。"""
    
    results = {
        "baseline": None,
        "combat": None,
        "improvement": {},
    }
    
    # 注意：这里是框架，实际使用需要修改 train_gat_cv.py 来保存逐样本预测
    # 目前只能手动从 predictions 日志中提取
    
    return results


def main():
    parser = argparse.ArgumentParser(description="对比站点效应：ComBat 前后")
    parser.add_argument("--baseline", type=Path, help="基线模型输出目录")
    parser.add_argument("--combat", type=Path, help="ComBat 调和后模型输出目录")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))
    args = parser.parse_args()
    
    # 加载站点 ID
    data = np.load(args.npz_path, allow_pickle=True)
    site_ids = data["site_ids"]
    
    print("\n" + "=" * 70)
    print("站点效应对比分析")
    print("=" * 70)
    
    print("\n📌 当前框架说明：")
    print("  - 要完整使用此脚本，需要在 train_gat_cv.py 中保存逐样本预测")
    print("  - 临时方案：手动从输出日志中提取 AUC 并对比")
    print("")
    
    if args.baseline:
        print(f"基线模型: {args.baseline}")
        metrics_file = args.baseline / "cv_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                baseline_data = json.load(f)
            print(f"  全局 AUC: {baseline_data['mean_auc']:.4f} ± {baseline_data['std_auc']:.4f}")
    
    if args.combat:
        print(f"\nComBat 调和: {args.combat}")
        metrics_file = args.combat / "cv_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                combat_data = json.load(f)
            print(f"  全局 AUC: {combat_data['mean_auc']:.4f} ± {combat_data['std_auc']:.4f}")
    
    print("\n💡 建议：")
    print("  1. 修改 train_gat_cv.py 的 evaluate() 函数以保存逐样本预测")
    print("  2. 在每个 fold 后保存: (labels, predictions, probabilities, site_ids)")
    print("  3. 重新运行训练，然后本脚本就能自动计算逐站点指标对比")


if __name__ == "__main__":
    main()
