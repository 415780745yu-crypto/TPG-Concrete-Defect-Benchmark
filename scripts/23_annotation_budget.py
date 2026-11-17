"""23: 标注预算曲线（新主图）。图像级 img100/200/500 带 3-seed 误差棒，zero-shot 下界、
full_ft / target_only 上界，source_only 为源域内参照线。

产物：results/tables/fig_annotation_budget.csv、results/fig_annotation_budget.png
"""
from __future__ import annotations

import csv
import numpy as np
import pandas as pd

from common import RESULTS

def main():
    cross = pd.read_csv(RESULTS / "crossdomain.csv")
    cross["tag"] = cross["tag"].astype(str)

    def m50(tag):
        hit = cross[cross["tag"] == tag]
        return float(hit["map50"].iloc[0]) if len(hit) else None

    zero = m50("zero_shot")
    full_ft = m50("full_ft")
    target_only = m50("target_only")
    source_only = m50("source_only")

    img = cross[cross["tag"].str.startswith("img", na=False)].copy()
    img["budget"] = img["tag"].str.extract(r"img(\d+)").astype(int)
    g = img.groupby("budget")["map50"].agg(mean="mean", std="std", n="count").reset_index()

    # 数据表（7 行：zero + 3 img + full_ft + target_only + source_only 参照）
    rows = [{"budget": 0, "setting": "zero_shot", "map50": round(zero, 4), "map50_std": "", "n_seeds": 1, "role": "泛化下界"}]
    for _, r in g.iterrows():
        rows.append({"budget": int(r["budget"]), "setting": f"img{int(r['budget'])}",
                     "map50": round(float(r["mean"]), 4), "map50_std": round(float(r["std"]), 4),
                     "n_seeds": int(r["n"]), "role": "图像级少样本"})
    rows.append({"budget": 2800, "setting": "full_ft", "map50": round(full_ft, 4), "map50_std": "", "n_seeds": 1, "role": "全量微调"})
    rows.append({"budget": 2800, "setting": "target_only", "map50": round(target_only, 4), "map50_std": "", "n_seeds": 1, "role": "目标域专属上界"})
    rows.append({"budget": "", "setting": "source_only", "map50": round(source_only, 4), "map50_std": "", "n_seeds": 1, "role": "源域内参考(3类,不同口径)"})

    with open(RESULTS / "tables" / "fig_annotation_budget.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = ["budget", "setting", "map50", "map50_std", "n_seeds", "role"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[write] results/tables/fig_annotation_budget.csv ({len(rows)} 行)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        pts = [(40, zero, 0), (100, g[g.budget == 100]["mean"].iloc[0], g[g.budget == 100]["std"].iloc[0]),
               (200, g[g.budget == 200]["mean"].iloc[0], g[g.budget == 200]["std"].iloc[0]),
               (500, g[g.budget == 500]["mean"].iloc[0], g[g.budget == 500]["std"].iloc[0])]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; errs = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=errs, fmt="-o", capsize=4, color="#1f77b4",
                    label="图像级少样本 (3 seeds)")
        ax.scatter([2800, 2800], [full_ft, target_only], marker="^", color=["#2ca02c", "#d62728"],
                   label=None, zorder=5)
        ax.annotate("full-FT", (2800, full_ft), textcoords="offset points", xytext=(6, -4), fontsize=8)
        ax.annotate("target-only", (2800, target_only), textcoords="offset points", xytext=(6, -16), fontsize=8)
        if source_only is not None:
            ax.axhline(source_only, color="#7f7f7f", linestyle="--", linewidth=1)
            ax.text(130, source_only + 0.012, f"source-only (S->S, 3-class, different scope) {source_only:.3f}",
                    fontsize=8, color="#555555")
        if zero is not None:
            ax.annotate(f"zero-shot {zero:.3f}", (40, zero), textcoords="offset points", xytext=(8, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([40, 100, 200, 500, 2800])
        ax.set_xticklabels(["0", "100", "200", "500", "2800"])
        ax.set_xlabel("annotation budget (number of annotated images, log scale)")
        ax.set_ylabel("mAP50 on DamCrack test (target domain)")
        ax.set_title("Cross-domain annotation budget curve")
        ax.grid(alpha=0.25)
        ax.set_xlim(30, 5000)
        ax.set_ylim(0, 0.8)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_annotation_budget.png", dpi=150)
        print("[write] results/fig_annotation_budget.png")
    except Exception as exc:
        print(f"[warn] 图未生成: {exc}")


if __name__ == "__main__":
    main()
