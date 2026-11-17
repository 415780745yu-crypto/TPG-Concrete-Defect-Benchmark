# -*- coding: utf-8 -*-
"""32: 方案 A（伪标签自训练）最终报告 —— 修正 x 轴口径问题。

问题：shot*（instance 预算）与 img*（图像预算）单位不同，混在同一 x 轴不严谨。
方案：主图用「真标注框数」(gt_boxes) 作共同自变量，两者统一为"标注框数量"这一可比单位。
     辅图保留 tier 顺序视图，并标注真:伪信噪比。

产物：
  - results/tables/fig_pl_delta_v2.csv
  - results/fig_pl_delta_vs_boxes.png   （主图：Δ vs 真标注框数）
  - results/fig_pl_delta_by_tier.png    （辅图：Δ vs 档位，标注信噪比）

运行：python scripts/32_pl_report_v2.py
"""
from __future__ import annotations

import csv

import pandas as pd

from common import RESULTS

ORDER = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]


def main() -> None:
    cross = pd.read_csv(RESULTS / "crossdomain.csv", dtype={"tag": str})
    comp = pd.read_csv(RESULTS / "pl_box_composition.csv")
    comp = comp.set_index("tier")

    rows = []
    for tier in ORDER:
        a = cross[cross["tag"] == f"{tier}_seed42"]
        p = cross[cross["tag"] == f"{tier}_seed42_pl"]
        if len(a) == 0 or len(p) == 0:
            continue
        gt = float(a["map50"].iloc[0])
        pl = float(p["map50"].iloc[0])
        rows.append({
            "tier": tier,
            "gt_boxes": int(comp.loc[tier, "gt_boxes"]),
            "pseudo_boxes": int(comp.loc[tier, "pseudo_boxes"]),
            "snr": round(float(comp.loc[tier, "pseudo_boxes"]) / float(comp.loc[tier, "gt_boxes"]), 2),
            "A_map50": round(gt, 4),
            "PL_map50": round(pl, 4),
            "delta": round(pl - gt, 4),
            "rel_gain_pct": round((pl - gt) / gt * 100, 1),
        })

    if not rows:
        print("[info] 尚无已评估档位")
        return

    dst = RESULTS / "tables" / "fig_pl_delta_v2.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {dst} ({len(rows)} 行)")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ---- 主图：Δ vs 真标注框数（可比单位）----
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.axhline(0, color="#888888", linestyle="--", linewidth=1.2, label="Δ = 0")
        ax.plot(df["gt_boxes"], df["delta"], "-o", color="#1f77b4", linewidth=2,
                markersize=7, label="A-PL − A (mAP50)")
        for _, r in df.iterrows():
            ax.annotate(f"{r['delta']:+.3f}", (r["gt_boxes"], r["delta"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlabel("number of ground-truth boxes in training set (log scale)")
        ax.set_ylabel("Δ mAP50 (pseudo-label − supervised)")
        ax.set_title("Pseudo-label gain vs. annotation size")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_pl_delta_vs_boxes.png", dpi=160)
        print("[write] results/fig_pl_delta_vs_boxes.png")

        # ---- 辅图：Δ vs 档位（标注信噪比）----
        fig2, ax2 = plt.subplots(figsize=(8.0, 4.8))
        xs = list(range(len(df)))
        ax2.bar(xs, df["delta"], color=["#2ca02c" if d > 0 else "#d62728" for d in df["delta"]],
                edgecolor="white", linewidth=0.6)
        ax2.axhline(0, color="#333333", linewidth=1)
        ymax = float(df["delta"].max())
        ax2.set_ylim(0, ymax * 1.28)
        for x, r in zip(xs, df.itertuples()):
            ax2.text(x, r.delta + ymax * 0.035, f"{r.delta:+.3f}",
                     ha="center", fontsize=9, fontweight="bold")
            ax2.text(x, ymax * 0.045, f"1:{r.snr:.0f}", ha="center",
                     fontsize=8, color="white", fontweight="bold")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(df["tier"], rotation=20)
        ax2.set_ylabel("Δ mAP50 (pseudo-label − supervised)")
        ax2.set_xlabel("annotation tier  (in-bar: pseudo:gt box ratio)")
        ax2.set_title("Pseudo-label gain by annotation tier")
        ax2.grid(alpha=0.25, axis="y")
        fig2.tight_layout()
        fig2.savefig(RESULTS / "fig_pl_delta_by_tier.png", dpi=160)
        print("[write] results/fig_pl_delta_by_tier.png")
    except Exception as exc:
        print(f"[warn] 图未生成: {exc}")


if __name__ == "__main__":
    main()
