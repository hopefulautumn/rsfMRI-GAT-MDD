# 🏥 rsfMRI-GAT 中的站点差异处理完全指南

---

## 问题背景

在多中心 fMRI 研究中，不同扫描仪（scanner）、参数设置、磁场强度等造成的**站点差异（site effect）** 往往比生物学信号还强。如果不妥善处理，模型可能学到的是"站点特异性信号"而非"MDD vs HC 的生物学区别"。

你的数据集中有多个站点（S01, S02...），所以必须系统性地控制这个混淆因素。

---

## 📋 当前方案总结

### ✅ 已实现（第 1-2 层防线）

| 层级 | 机制 | 代码位置 | 效果 |
|------|------|---------|------|
| **数据集构建** | 自动提取站点 ID（从文件名）| `build_fc_dataset.py` | 便于后续分析 |
| **交叉验证** | StratifiedGroupKFold 按站点分层 | `train_gat_cv.py` L325 | 防止"信息泄漏"——不同站点样本不会同时在 train/val |
| **数据调和** | ComBat 方法消除站点参数差异 | `site_harmonization.py` | 参数级别的站点消除（推荐使用！） |

### ❌ 可增强（第 3-4 层防线）

| 层级 | 机制 | 何时使用 |
|------|------|---------|
| **模型对抗** | Domain Adversarial Training | 需要超强鲁棒性时 |
| **验证指标** | Site-balanced metrics 评估 | 每次实验都应该做 |

---

## 🚀 快速开始：4 步处理流程

### **第 1 步：用 ComBat 调和 FC 矩阵（最重要！）**

```bash
# 需要先安装依赖
pip install neurocombat-sklearn

# 生成调和后的 FC 数据集
python -c "
from pathlib import Path
from src.site_harmonization import CombatHarmonizer
import numpy as np

data = np.load('processed/rest_meta_mdd_fc.npz', allow_pickle=True)
fc_train = data['fc_matrices']
site_ids = data['site_ids']
labels = data['labels']

harmonizer = CombatHarmonizer()
fc_harmonized = harmonizer.fit_transform_train(fc_train, site_ids)

Path('processed').mkdir(exist_ok=True)
np.savez_compressed(
    'processed/rest_meta_mdd_fc_combat.npz',
    fc_matrices=fc_harmonized,
    labels=labels,
    group_names=data['group_names'],
    site_ids=site_ids,
    group_order=data['group_order'],
)
print('✅ ComBat 调和完成！')
"
```

**原理**：
- 从原始 FC 矩阵中分离出"站点特异的均值和方差"
- 标准化到全局空间，消除站点"系统偏移"
- 保留生物学信号的相对关系

### **第 2 步：用调和后的数据训练，启用分组 CV**

```bash
python src/train_gat_cv.py \
    --npz-path processed/rest_meta_mdd_fc_combat.npz \
    --output-dir outputs/gat_cv_combat_v1 \
    --n-splits 5 \
    --seed 42 \
    --site-harmonization combat
```

**关键参数说明**：
- `--npz-path` 指向调和后的文件
- `--site-harmonization combat`：在每个 fold 内再做一轮 ComBat（保险做法）
- 注意：ComBat 需要验证集的所有站点都出现在训练集中（StratifiedGroupKFold 保证）

### **第 3 步：评估站点均衡性**

```bash
python src/site_balance_validator.py \
    --metrics-json outputs/gat_cv_combat_v1/cv_metrics.json \
    --output-report outputs/gat_cv_combat_v1/site_balance_report.json
```

**报告包含**：
- 全局 AUC / F1 等指标
- **每个站点的 AUC**（关键！应该是"相近"的）
- 站点间差异的方差与建议

**好的迹象**：
```
AUC std = 0.05   ✅ 表示模型对所有站点均匀有效
AUC std = 0.15   ⚠️  表示站点偏差仍很强，需要增强调和
AUC std = 0.30   ❌ 表示模型可能在学站点特异性
```

### **第 4 步：对比实验（推荐）**

```bash
# 无调和基线
python src/train_gat_cv.py \
    --npz-path processed/rest_meta_mdd_fc.npz \
    --output-dir outputs/gat_cv_baseline \
    --site-harmonization none

# 有调和版本
python src/train_gat_cv.py \
    --npz-path processed/rest_meta_mdd_fc_combat.npz \
    --output-dir outputs/gat_cv_combat \
    --site-harmonization combat
```

对比两版本的 site_balance_report，若 ComBat 版本的 AUC_std 显著下降，说明调和有效。

---

## 📊 关键指标一览表

| 指标 | 含义 | 目标值 |
|------|------|--------|
| **全局 AUC** | 整体分类性能 | > 0.70 |
| **AUC_std（逐站点）** | 站点间差异 | < 0.10（很好）; < 0.15（可接受）|
| **AUC_range** | 最高/最低站点 AUC | 差异 < 0.15 |

---

## 🐛 常见问题

### Q: ComBat 报错"未见过的站点"
**A**: 用 `--site-harmonization none` 或检查 CV 配置。StratifiedGroupKFold 应该已解决此问题。

### Q: ComBat 后性能反而下降
**A**: 检查样本量——若某站点 < 5 个样本，参数估计不稳定。考虑合并小站点或用 `--site-harmonization none`。

### Q: 怎么判断 ComBat 有没有生效？
**A**: 比较调和前后 `site_balance_report` 中的 `auc_std`，应显著下降（如从 0.25 → 0.08）。

---

## 📁 核心文件速查

| 文件 | 用途 |
|------|------|
| `src/build_fc_dataset.py` | 数据构建：自动提取 site_ids |
| `src/site_harmonization.py` | ComBat 实现 |
| `src/train_gat_cv.py` | 主训练脚本（已集成 ComBat + StratifiedGroupKFold）|
| `src/site_balance_validator.py` | 站点均衡性评估 |

---

## ✅ 最佳实践清单

- [ ] 原始 FC 用 ComBat 调和
- [ ] 用 StratifiedGroupKFold（自动选择）
- [ ] 每次实验输出 `site_balance_report`
- [ ] 论文中报告"站点调和方法"和"逐站点 AUC"
- [ ] 记录完整的训练参数（seed, --site-harmonization 等）

