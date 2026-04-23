from __future__ import annotations

"""将 important_edges.csv 映射为脑区名称并做网络/系统统计。

把模型输出的边级结果翻译成更容易理解的脑区名、功能系统名和系统级统计量。

输出：
important_edges_mapped.csv: 每条边的脑区名与系统归属。
network_pair_stats.csv: 系统内/系统间连接统计汇总。


- 输入是 explain_edges.py / explain_edges_split.py 导出的边表。
- 输出是可以直接放入论文附表或结果分析的脑区映射表与系统统计表。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# AAL116 标准顺序（按 1-based 列出，代码中会映射为 0-based 索引）。
#
# 这个列表非常关键：
# - FC 矩阵是 116x116
# - 矩阵中的第 0 个位置并不直接告诉我们“它是什么脑区”
# - 必须借助一个固定的图谱顺序，才能把 roi 编号翻译成具体脑区名称
#
# 这里采用的是 AAL116 的常见命名顺序。
# 也就是说：roi_u=0 并不是“第 0 个脑区”这种抽象概念，
# 而是会被映射到 AAL116_LABELS[0] 对应的具体解剖名称。
AAL116_LABELS = [
    "Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R", "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
    "Frontal_Mid_L", "Frontal_Mid_R", "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R", "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R", "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R", "Rolandic_Oper_L", "Rolandic_Oper_R",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R", "Olfactory_L", "Olfactory_R", "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R", "Rectus_L", "Rectus_R", "Insula_L", "Insula_R", "Cingulum_Ant_L",
    "Cingulum_Ant_R", "Cingulum_Mid_L", "Cingulum_Mid_R", "Cingulum_Post_L", "Cingulum_Post_R", "Hippocampus_L",
    "Hippocampus_R", "ParaHippocampal_L", "ParaHippocampal_R", "Amygdala_L", "Amygdala_R", "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R", "Lingual_L", "Lingual_R", "Occipital_Sup_L", "Occipital_Sup_R", "Occipital_Mid_L",
    "Occipital_Mid_R", "Occipital_Inf_L", "Occipital_Inf_R", "Fusiform_L", "Fusiform_R", "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R", "Parietal_Inf_L", "Parietal_Inf_R", "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R", "Precuneus_L", "Precuneus_R", "Paracentral_Lobule_L", "Paracentral_Lobule_R", "Caudate_L",
    "Caudate_R", "Putamen_L", "Putamen_R", "Pallidum_L", "Pallidum_R", "Thalamus_L", "Thalamus_R", "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R", "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R", "Temporal_Mid_L", "Temporal_Mid_R",
    "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R", "Temporal_Inf_L", "Temporal_Inf_R", "Cerebelum_Crus1_L", "Cerebelum_Crus1_R",
    "Cerebelum_Crus2_L", "Cerebelum_Crus2_R", "Cerebelum_3_L", "Cerebelum_3_R", "Cerebelum_4_5_L", "Cerebelum_4_5_R",
    "Cerebelum_6_L", "Cerebelum_6_R", "Cerebelum_7b_L", "Cerebelum_7b_R", "Cerebelum_8_L", "Cerebelum_8_R",
    "Cerebelum_9_L", "Cerebelum_9_R", "Cerebelum_10_L", "Cerebelum_10_R", "Vermis_1_2", "Vermis_3", "Vermis_4_5",
    "Vermis_6", "Vermis_7", "Vermis_8", "Vermis_9", "Vermis_10",
]


def infer_system(roi_name: str) -> str:
    """按 AAL 解剖命名推断粗粒度系统。

    参数
    ----
    roi_name:
        单个 ROI 的 AAL 解剖名称，例如 "Frontal_Sup_L"、"Insula_R"。

    返回
    ----
    str
        该 ROI 所属的粗粒度功能/解剖系统名。

    说明
    ----
    这里不是做精细的功能网络分区，而是做“粗粒度归类”，
    目的是把 116 个具体脑区压缩成少数几个更容易解释的系统：
    - Frontal（额叶）
    - Parietal（顶叶）
    - Temporal（颞叶）
    - Occipital（枕叶）
    - Limbic（边缘系统）
    - Subcortical（皮层下）
    - Cerebellum（小脑）
    - Other（无法归类时兜底）

    让边级结果不仅停留在“ROI 63 连 ROI 34 很重要”，
    还能上升到“顶叶-边缘系统连接很重要”这种更容易写论文的层次。
    """

    # 小脑相关命名。
    if roi_name.startswith("Cerebelum") or roi_name.startswith("Vermis"):
        return "Cerebellum"
    # 皮层下核团：尾状核、壳核、苍白球、丘脑。
    if roi_name.startswith(("Caudate", "Putamen", "Pallidum", "Thalamus")):
        return "Subcortical"
    # 边缘系统：海马、海马旁回、杏仁核、扣带回、岛叶、嗅觉区。
    if roi_name.startswith(("Hippocampus", "ParaHippocampal", "Amygdala", "Cingulum", "Insula", "Olfactory")):
        return "Limbic"
    # 颞叶与听觉相关脑区。
    if roi_name.startswith(("Temporal", "Heschl")):
        return "Temporal"
    # 枕叶及视觉相关脑区。
    if roi_name.startswith(("Occipital", "Calcarine", "Cuneus", "Lingual", "Fusiform")):
        return "Occipital"
    # 顶叶、中央后回、缘上回、角回、楔前叶、旁中央小叶等。
    if roi_name.startswith(("Parietal", "Postcentral", "SupraMarginal", "Angular", "Precuneus", "Paracentral")):
        return "Parietal"
    # 额叶、中央前回、Rolandic 区、辅助运动区、直回等。
    if roi_name.startswith(("Frontal", "Precentral", "Rolandic", "Supp_Motor", "Rectus")):
        return "Frontal"
    # 兜底项：如果名称不落入上述任何规则，就标记为 Other。
    return "Other"


def load_edges(csv_path: Path) -> list[dict[str, str]]:
    """读取上游解释脚本导出的边表。

    这个 CSV 的每一行通常代表一条重要连接边，至少包含：
    - roi_u
    - roi_v
    - mean_attention
    - support_count

    返回值是字典列表，方便后面按列名访问。
    """

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """主流程：读边表 -> 映射脑区名 -> 汇总系统统计 -> 导出 CSV。"""

    # argparse 的作用：把命令行中的输入/输出路径解析成 Python 对象。
    parser = argparse.ArgumentParser(description="Map important edges to AAL116 names and summarize systems")
    # 输入文件：由 explain_edges.py 或 explain_edges_split.py 生成的边级重要性表。
    parser.add_argument("--input-csv", type=Path, default=Path("outputs/gat_cv_formal/important_edges.csv"))
    # 输出文件 1：边级映射表，记录“边的两个端点分别是什么脑区”。
    parser.add_argument(
        "--output-mapped-csv",
        type=Path,
        default=Path("outputs/gat_cv_formal/important_edges_mapped.csv"),
    )
    # 输出文件 2：系统级统计表，按“系统-系统”对边进行汇总。
    parser.add_argument(
        "--output-network-csv",
        type=Path,
        default=Path("outputs/gat_cv_formal/network_pair_stats.csv"),
    )
    args = parser.parse_args()

    # 读取输入边表。
    rows = load_edges(args.input_csv)

    # mapped_rows：逐边的详细映射结果，最后写入 important_edges_mapped.csv。
    # pair_score_sum：按系统对累加 mean_attention。
    # pair_support_sum：按系统对累加 support_count，代表该系统对被“多少次样本证实”。
    # pair_edge_count：按系统对统计有多少条边落入该系统组合。
    mapped_rows: list[dict[str, object]] = []
    pair_score_sum: dict[tuple[str, str], float] = defaultdict(float)
    pair_support_sum: dict[tuple[str, str], int] = defaultdict(int)
    pair_edge_count: dict[tuple[str, str], int] = defaultdict(int)

    # 遍历每一条边，完成“ROI 编号 -> AAL 名称 -> 系统归类”的转换。
    for rank, row in enumerate(rows, start=1):
        # roi_u / roi_v 是图中的两个端点编号。
        roi_u = int(row["roi_u"])
        roi_v = int(row["roi_v"])
        # mean_attention 表示这条边在解释脚本里聚合后的平均注意力分数。
        mean_attention = float(row["mean_attention"])
        # support_count 表示这条边在多少次统计中出现过。
        support_count = int(row["support_count"])

        # 安全检查：防止输入 CSV 的 ROI 索引超出 AAL116 范围。
        if not (0 <= roi_u < len(AAL116_LABELS) and 0 <= roi_v < len(AAL116_LABELS)):
            raise ValueError(f"ROI index out of range for AAL116: ({roi_u}, {roi_v})")

        # 把整数索引翻译成可读的脑区名。
        roi_u_name = AAL116_LABELS[roi_u]
        roi_v_name = AAL116_LABELS[roi_v]
        # 再把脑区名归并到粗粒度系统。
        system_u = infer_system(roi_u_name)
        system_v = infer_system(roi_v_name)

        # 无向边统计：把 (A, B) 与 (B, A) 看作同一系统对。
        # sorted 的作用就是保证顺序固定，避免重复统计。
        pair = tuple(sorted((system_u, system_v)))
        pair_score_sum[pair] += mean_attention
        pair_support_sum[pair] += support_count
        pair_edge_count[pair] += 1

        # 记录每一条边的完整解释信息。
        mapped_rows.append(
            {
                "rank": rank,
                "roi_u": roi_u,
                "roi_v": roi_v,
                "roi_u_name": roi_u_name,
                "roi_v_name": roi_v_name,
                "system_u": system_u,
                "system_v": system_v,
                "system_pair": f"{pair[0]}-{pair[1]}",
                "mean_attention": mean_attention,
                "support_count": support_count,
            }
        )

    # 写出边级映射表。
    # 这个文件适合人工查看：
    # - 某条边具体连接的是哪两个脑区
    # - 这两个脑区分别属于什么系统
    # - 这条边的重要性分数是多少
    args.output_mapped_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapped_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "roi_u",
                "roi_v",
                "roi_u_name",
                "roi_v_name",
                "system_u",
                "system_v",
                "system_pair",
                "mean_attention",
                "support_count",
            ],
        )
        writer.writeheader()
        writer.writerows(mapped_rows)

    # 从逐边结果进一步上升到“系统对”层面。
    # 例如：Frontal-Parietal、Limbic-Parietal、Cerebellum-Parietal 等。
    network_rows: list[dict[str, object]] = []
    for pair, score_sum in pair_score_sum.items():
        # 一共有多少条边落在这个系统对里。
        edge_count = pair_edge_count[pair]
        # 该系统对所有边的 support_count 总和。
        support_total = pair_support_sum[pair]
        network_rows.append(
            {
                "system_pair": f"{pair[0]}-{pair[1]}",
                # 该系统对中边的条数。
                "edge_count": edge_count,
                # 该系统对中所有边的平均注意力。
                "mean_attention_avg": score_sum / edge_count,
                # 该系统对中 support_count 的总和。
                "support_count_total": support_total,
                # 每条边平均 support_count，反映稳定性。
                "support_count_avg": support_total / edge_count,
                # 加权分数：把注意力强度和稳定性结合起来。
                # 这个值越高，表示“既强又稳”的系统对越值得关注。
                "weighted_score": score_sum * support_total,
            }
        )

    # 按 weighted_score 从高到低排序，优先展示最值得关注的系统对。
    network_rows.sort(key=lambda x: float(x["weighted_score"]), reverse=True)

    # 写出系统级统计表。
    # 这个文件更适合做结果总结：
    # 看哪些系统之间的连接最重要，是否存在明显的网络模式偏向。
    with args.output_network_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system_pair",
                "edge_count",
                "mean_attention_avg",
                "support_count_total",
                "support_count_avg",
                "weighted_score",
            ],
        )
        writer.writeheader()
        writer.writerows(network_rows)

    # 终端提示输出完成。
    print(f"Saved mapped edges: {args.output_mapped_csv}")
    print(f"Saved network stats: {args.output_network_csv}")


if __name__ == "__main__":
    main()
