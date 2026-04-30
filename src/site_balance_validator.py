"""站点差异验证模块 - 评估模型是否真正消除了站点偏差。

核心思想：
- 一个好的模型应该对所有站点都有一致的预测性能。
- 如果某个站点的 AUC 显著高于或低于其他站点，说明模型在学习站点特异性而非生物学信号。
- 对抗训练（domain adversarial）可以进一步削弱这种依赖。
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score


class SiteMetrics(NamedTuple):
    """单个站点的性能指标。"""
    site_id: str
    n_samples: int
    accuracy: float
    auc: float
    f1: float
    precision: float
    recall: float


@dataclass
class SiteBalanceReport:
    """全量站点均衡性评估报告。"""
    
    overall_metrics: dict
    per_site_metrics: list[SiteMetrics]
    site_variance: dict  # 每个指标在站点间的方差/标准差
    recommendations: list[str]
    
    def to_dict(self) -> dict:
        return {
            "overall_metrics": self.overall_metrics,
            "per_site_metrics": [
                {
                    "site_id": m.site_id,
                    "n_samples": m.n_samples,
                    "accuracy": float(m.accuracy),
                    "auc": float(m.auc),
                    "f1": float(m.f1),
                    "precision": float(m.precision),
                    "recall": float(m.recall),
                }
                for m in self.per_site_metrics
            ],
            "site_variance": self.site_variance,
            "recommendations": self.recommendations,
        }


def compute_site_balanced_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    site_ids: np.ndarray,
) -> SiteBalanceReport:
    """计算全局和逐站点的性能指标，检测站点间不均衡。

    参数：
        labels: [N] 真实标签 (0 或 1)。
        predictions: [N] 模型预测标签 (0 或 1)。
        probabilities: [N] 正类概率 (0~1)。
        site_ids: [N] 每个样本的站点标签。

    返回：
        SiteBalanceReport 对象，包含全局指标、逐站点指标、差异统计和建议。
    """

    # 全局指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    overall = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "n_total": int(len(labels)),
    }

    # 逐站点计算
    unique_sites = np.unique(site_ids)
    per_site = []
    per_site_auc = []
    per_site_f1 = []
    
    for site in unique_sites:
        mask = site_ids == site
        site_labels = labels[mask]
        site_preds = predictions[mask]
        site_probs = probabilities[mask]
        n = np.sum(mask)
        
        if len(np.unique(site_labels)) > 1:
            site_auc = roc_auc_score(site_labels, site_probs)
            per_site_auc.append(site_auc)
        else:
            site_auc = 0.0
        
        site_metrics = SiteMetrics(
            site_id=str(site),
            n_samples=int(n),
            accuracy=float(accuracy_score(site_labels, site_preds)),
            auc=site_auc,
            f1=float(f1_score(site_labels, site_preds, zero_division=0)),
            precision=float(precision_score(site_labels, site_preds, zero_division=0)),
            recall=float(recall_score(site_labels, site_preds, zero_division=0)),
        )
        per_site.append(site_metrics)

    # 计算站点间差异
    if per_site_auc:
        auc_variance = float(np.var(per_site_auc))
        auc_std = float(np.std(per_site_auc))
        auc_range = (float(np.min(per_site_auc)), float(np.max(per_site_auc)))
    else:
        auc_variance = 0.0
        auc_std = 0.0
        auc_range = (0.0, 0.0)

    if per_site_f1:
        f1_variance = float(np.var([m.f1 for m in per_site]))
        f1_std = float(np.std([m.f1 for m in per_site]))
    else:
        f1_variance = 0.0
        f1_std = 0.0

    site_variance = {
        "auc_variance": auc_variance,
        "auc_std": auc_std,
        "auc_range": auc_range,
        "f1_variance": f1_variance,
        "f1_std": f1_std,
    }

    # 生成建议
    recommendations = []
    if auc_std > 0.15:
        recommendations.append(
            "⚠️ 站点间 AUC 差异很大（std={:.3f}）。"
            "建议：激活 ComBat 调和或尝试对抗域适应训练。".format(auc_std)
        )
    elif auc_std > 0.08:
        recommendations.append(
            "⚠️ 站点间 AUC 差异中等（std={:.3f}）。"
            "可考虑增强数据调和。".format(auc_std)
        )
    else:
        recommendations.append(
            "✅ 站点间 AUC 差异较小（std={:.3f}），模型表现良好。".format(auc_std)
        )

    return SiteBalanceReport(
        overall_metrics=overall,
        per_site_metrics=per_site,
        site_variance=site_variance,
        recommendations=recommendations,
    )


def print_site_balance_report(report: SiteBalanceReport) -> None:
    """美化打印站点均衡性报告。"""

    print("\n" + "=" * 70)
    print("站点均衡性评估报告")
    print("=" * 70)

    print("\n[全局指标]")
    for key, val in report.overall_metrics.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")

    print("\n[逐站点指标]")
    print(f"{'Site':6s} {'N':8s} {'AUC':8s} {'F1':8s} {'Acc':8s} {'Prec':8s} {'Rec':8s}")
    print("-" * 56)
    for metric in report.per_site_metrics:
        print(
            f"{metric.site_id:6s} {metric.n_samples:8d} "
            f"{metric.auc:8.4f} {metric.f1:8.4f} {metric.accuracy:8.4f} "
            f"{metric.precision:8.4f} {metric.recall:8.4f}"
        )

    print("\n[站点间差异统计]")
    print(f"  AUC std    : {report.site_variance['auc_std']:.4f}")
    print(f"  AUC range  : {report.site_variance['auc_range']}")
    print(f"  F1 std     : {report.site_variance['f1_std']:.4f}")

    print("\n[建议]")
    for rec in report.recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 70)


def save_site_balance_report(
    report: SiteBalanceReport,
    output_path: Path,
) -> None:
    """保存报告为 JSON。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[INFO] 报告已保存: {output_path}")


if __name__ == "__main__":
    # 示例使用
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n)
    predictions = np.random.randint(0, 2, n)
    probabilities = np.random.rand(n)
    sites = np.random.choice(["S01", "S02", "S03"], n)

    report = compute_site_balanced_metrics(labels, predictions, probabilities, sites)
    print_site_balance_report(report)
