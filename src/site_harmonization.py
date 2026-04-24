from __future__ import annotations

"""站点层面的特征校正工具。"""

from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np


def _patch_neurocombat_onehotencoder() -> None:
    """兼容 neurocombat-sklearn 与新版本 scikit-learn 的 OneHotEncoder 参数差异。"""

    import sklearn.preprocessing as sk_pre
    from neurocombat_sklearn import neurocombat_sklearn as nc_module

    sig = inspect.signature(sk_pre.OneHotEncoder.__init__)
    if "sparse" in sig.parameters:
        # 旧版 sklearn 无需补丁。
        return

    def compat_one_hot_encoder(*args, **kwargs):  # type: ignore[no-untyped-def]
        if "sparse" in kwargs and "sparse_output" not in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return sk_pre.OneHotEncoder(*args, **kwargs)

    nc_module.OneHotEncoder = compat_one_hot_encoder


@dataclass
class CombatHarmonizer:
    """基于 ComBat 的 FC 特征校正器。

    注意：transform 时要求站点标签必须在 fit 时见过。
    """

    n_rois: int | None = None
    triu_idx: tuple[np.ndarray, np.ndarray] | None = None
    seen_sites: set[str] | None = None
    site_to_code: dict[str, int] | None = None
    model: Any | None = None

    def _encode_sites(self, site_data: np.ndarray) -> np.ndarray:
        if self.site_to_code is None:
            raise RuntimeError("站点编码映射未初始化")
        flat = np.asarray(site_data, dtype=object).astype(str).reshape(-1)
        unseen = sorted(set(flat.tolist()) - set(self.site_to_code.keys()))
        if unseen:
            raise ValueError(
                f"ComBat transform 检测到未见过的站点: {unseen}。"
                "请在不含未见站点的评估协议中使用，或关闭 ComBat。"
            )
        codes = np.asarray([self.site_to_code[s] for s in flat], dtype=np.int64)
        return codes.reshape(-1, 1)

    def _flatten_upper(self, fc_matrices: np.ndarray) -> np.ndarray:
        if self.triu_idx is None:
            raise RuntimeError("Harmonizer 尚未初始化上三角索引")
        i, j = self.triu_idx
        return fc_matrices[:, i, j]

    def _restore_upper(self, features: np.ndarray) -> np.ndarray:
        if self.triu_idx is None or self.n_rois is None:
            raise RuntimeError("Harmonizer 尚未初始化形状信息")
        i, j = self.triu_idx
        n = features.shape[0]
        out = np.zeros((n, self.n_rois, self.n_rois), dtype=np.float32)
        out[:, i, j] = features
        out[:, j, i] = features
        return out

    def fit(self, fc_train: np.ndarray, site_train: np.ndarray) -> "CombatHarmonizer":
        try:
            from neurocombat_sklearn import CombatModel
            _patch_neurocombat_onehotencoder()
        except ImportError as exc:
            raise ImportError(
                "未安装 neurocombat-sklearn。请先执行: uv add neurocombat-sklearn"
            ) from exc

        if fc_train.ndim != 3 or fc_train.shape[1] != fc_train.shape[2]:
            raise ValueError(f"fc_train 形状非法: {fc_train.shape}，应为 [N, R, R]")

        self.n_rois = int(fc_train.shape[1])
        self.triu_idx = np.triu_indices(self.n_rois, k=1)
        site_train_str = np.asarray(site_train, dtype=object).astype(str).reshape(-1)
        unique_sites = sorted(set(site_train_str.tolist()))
        self.site_to_code = {site: idx for idx, site in enumerate(unique_sites)}
        self.seen_sites = set(unique_sites)
        site_train_code = np.asarray([self.site_to_code[s] for s in site_train_str], dtype=np.int64).reshape(-1, 1)

        x_train = self._flatten_upper(fc_train).astype(np.float64)
        model = CombatModel()
        model.fit(x_train, site_train_code)
        self.model = model
        return self

    def transform(self, fc_data: np.ndarray, site_data: np.ndarray) -> np.ndarray:
        if self.model is None or self.seen_sites is None:
            raise RuntimeError("请先调用 fit")

        site_data_code = self._encode_sites(site_data)

        x_data = self._flatten_upper(fc_data).astype(np.float64)
        x_h = self.model.transform(x_data, site_data_code)
        return self._restore_upper(np.asarray(x_h, dtype=np.float32))

    def fit_transform_train(self, fc_train: np.ndarray, site_train: np.ndarray) -> np.ndarray:
        self.fit(fc_train, site_train)
        return self.transform(fc_train, site_train)
