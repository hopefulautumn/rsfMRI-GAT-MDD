from __future__ import annotations

"""传统机器学习基线（CV）训练脚本。

与 GAT 使用同一份 FC 数据，支持：
- StratifiedKFold
- StratifiedGroupKFold（按站点分组）

默认模型：SVM、Logistic Regression、Random Forest。
输入特征：每个样本 FC 矩阵上三角（不含对角）向量。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fc_to_graph_dataset import load_fc_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline ML models with CV")
    parser.add_argument("--npz-path", type=Path, default=Path("processed/rest_meta_mdd_fc.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_cv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--cv-mode",
        type=str,
        default="auto",
        choices=["auto", "grouped", "stratified"],
        help="auto=prefer grouped if site_ids available",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--models",
        type=str,
        default="svm,logreg,rf",
        help="comma-separated: svm,logreg,rf",
    )
    return parser.parse_args()


def flatten_fc_upper(fc_matrices: np.ndarray) -> np.ndarray:
    """把 [N, R, R] FC 转为 [N, P] 上三角特征。"""

    if fc_matrices.ndim != 3 or fc_matrices.shape[1] != fc_matrices.shape[2]:
        raise ValueError(f"fc_matrices 形状非法: {fc_matrices.shape}")
    r = fc_matrices.shape[1]
    iu = np.triu_indices(r, k=1)
    return fc_matrices[:, iu[0], iu[1]]


def build_models(selected: list[str], seed: int) -> dict[str, Any]:
    all_models: dict[str, Any] = {
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    unknown = [m for m in selected if m not in all_models]
    if unknown:
        raise ValueError(f"未知模型: {unknown}，可选: {list(all_models.keys())}")

    return {m: all_models[m] for m in selected}


def get_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    # 兜底：无分数输出时退化为0/1预测。
    return model.predict(x).astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else 0.0,
    }


def empty_metrics() -> dict[str, float]:
    return {
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "auc": float("nan"),
    }


def main() -> None:
    args = parse_args()

    fc_matrices, labels, _, site_ids = load_fc_npz(args.npz_path)
    if args.max_samples > 0:
        fc_matrices = fc_matrices[: args.max_samples]
        labels = labels[: args.max_samples]
        site_ids = site_ids[: args.max_samples]

    x = flatten_fc_upper(fc_matrices)
    y = np.asarray(labels, dtype=np.int64)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "当前样本仅包含单一类别，无法训练分类器。"
            "请增大 --max-samples 或不使用该参数。"
        )

    unique_sites = np.unique(site_ids)
    has_valid_site_ids = len(unique_sites) > 1 and not np.all(site_ids == "UNK")

    if args.cv_mode == "grouped" and not has_valid_site_ids:
        raise ValueError(
            "--cv-mode grouped 需要有效站点信息，请先重建包含 site_ids 的 npz"
        )

    use_grouped_cv = (args.cv_mode == "grouped") or (
        args.cv_mode == "auto" and has_valid_site_ids
    )

    if use_grouped_cv:
        splitter = StratifiedGroupKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed,
        )
        split_iter = splitter.split(np.zeros(len(y)), y, groups=site_ids)
        print(
            f"Using StratifiedGroupKFold | cv_mode={args.cv_mode} | "
            f"n_splits={args.n_splits} | n_sites={len(unique_sites)}"
        )
    else:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.zeros(len(y)), y)
        reason = "forced by --cv-mode stratified" if args.cv_mode == "stratified" else "site_ids unavailable"
        print(f"Using StratifiedKFold | cv_mode={args.cv_mode} | reason={reason}")

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = build_models(selected_models, seed=args.seed)

    print(f"Loaded {len(x)} samples | feature_dim={x.shape[1]} | models={list(models.keys())}")

    fold_indices = list(split_iter)
    all_results: dict[str, list[dict[str, float]]] = {name: [] for name in models}

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"\n===== Fold {fold_id}/{args.n_splits} =====")

        for name, model in models.items():
            if len(np.unique(y_train)) < 2:
                metrics = empty_metrics()
                all_results[name].append(metrics)
                print(f"[{name}] skipped: train fold has only one class")
                continue

            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            y_score = get_scores(model, x_val)
            metrics = compute_metrics(y_val, y_pred, y_score)
            all_results[name].append(metrics)
            print(
                f"[{name}] auc={metrics['auc']:.4f} | f1={metrics['f1']:.4f} | "
                f"acc={metrics['accuracy']:.4f}"
            )

    summary: dict[str, Any] = {
        "cv_splitter": splitter.__class__.__name__,
        "n_sites": int(len(unique_sites)),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "models": {},
    }

    for name, fold_metrics in all_results.items():
        summary["models"][name] = {
            "mean_accuracy": float(np.nanmean([m["accuracy"] for m in fold_metrics])),
            "mean_f1": float(np.nanmean([m["f1"] for m in fold_metrics])),
            "mean_auc": float(np.nanmean([m["auc"] for m in fold_metrics])),
            "std_accuracy": float(np.nanstd([m["accuracy"] for m in fold_metrics])),
            "std_f1": float(np.nanstd([m["f1"] for m in fold_metrics])),
            "std_auc": float(np.nanstd([m["auc"] for m in fold_metrics])),
            "fold_metrics": fold_metrics,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "baseline_cv_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Baseline Summary =====")
    for name in models:
        m: dict[str, Any] = summary["models"][name]
        print(
            f"{name}: AUC={m['mean_auc']:.4f}±{m['std_auc']:.4f} | "
            f"F1={m['mean_f1']:.4f}±{m['std_f1']:.4f} | "
            f"Acc={m['mean_accuracy']:.4f}±{m['std_accuracy']:.4f}"
        )
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
