# -*- coding: utf-8 -*-
"""37: 重绘图 4 —— 伪标签净增益随标注量的变化（三组对照）。

图 4 两个面板：
  (a) 三组对照（A / A-PL / repeat-A）的 mAP50 对真标注框数（对数横轴）
  (b) 净增益（A-PL − repeat-A）按档位的柱状图，标注临界区间 80–157 框

数据：results/crossdomain.csv 中 seed42/43/44 的 <tier>_seed<s>、_pl、_rpt
统计：三个标注抽样种子的均值 ± 总体标准差（ddof=0），与表 3 表注一致

产物：
  - results/fig4_pseudo_label_3arms.png
  - results/tables/fig4_data.csv

运行：python scripts/37_plot_fig4.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from common import RESULTS

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
SEEDS = (42, 43, 44)
GT_BOXES = {"shot10": 71, "shot20": 80, "shot30": 157, "shot50": 272,
            "img100": 1161, "img200": 2276, "img500": 5593}
# 真:伪 框比（与表 3 一致）
RATIO = {"shot10": "1:13", "shot20": "1:21", "shot30": "1:100", "shot50": "1:68",
         "img100": "1:15", "img200": "1:10", "img500": "1:4"}


def load() -> dict[str, float]:
    with open(RESULTS / "crossdomain.csv", "r", newline="", encoding="utf-8-sig") as f:
        idx = {}
        for r in csv.DictReader(f):
            idx.setdefault(r["tag"].strip(), r)
    out = {}
    for tag, r in idx.items():
        v = (r.get("map50") or "").strip()
        if v:
            out[tag] = float(v)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def sd0(xs):
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def main() -> None:
    data = load()
    rows = []
    for t in TIERS:
        A = [data[f"{t}_seed{s}"] for s in SEEDS if f"{t}_seed{s}" in data]
        P = [data[f"{t}_seed{s}_pl"] for s in SEEDS if f"{t}_seed{s}_pl" in data]
        R = [data[f"{t}_seed{s}_rpt"] for s in SEEDS if f"{t}_seed{s}_rpt" in data]
        if not (A and P and R):
            continue
        net = [p - r for p, r in zip(P, R)]
        rows.append({
            "tier": t, "gt_boxes": GT_BOXES[t], "ratio": RATIO[t],
            "A_mean": round(mean(A), 4), "A_sd": round(sd0(A), 4),
            "APL_mean": round(mean(P), 4), "APL_sd": round(sd0(P), 4),
            "RpT_mean": round(mean(R), 4), "RpT_sd": round(sd0(R), 4),
            "net_mean": round(mean(net), 4), "net_sd": round(sd0(net), 4),
            "n_seeds": len(A),
        })

    out_csv = RESULTS / "tables" / "fig4_data.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6))

        xs = [r["gt_boxes"] for r in rows]

        # ---------- (a) 三组对照 ----------
        ax1.errorbar(xs, [r["A_mean"] for r in rows], yerr=[r["A_sd"] for r in rows],
                     fmt="-o", capsize=3, lw=1.8, ms=6, color="#7f7f7f",
                     label="A  (ground truth only)")
        ax1.errorbar(xs, [r["APL_mean"] for r in rows], yerr=[r["APL_sd"] for r in rows],
                     fmt="-s", capsize=3, lw=1.8, ms=6, color="#1f77b4",
                     label="A-PL  (ground truth + pseudo-labels)")
        ax1.errorbar(xs, [r["RpT_mean"] for r in rows], yerr=[r["RpT_sd"] for r in rows],
                     fmt="-^", capsize=3, lw=1.8, ms=6, color="#d62728",
                     label="repeat-A  (ground truth repeated to equal size)")
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of ground-truth boxes in the training set (log scale)")
        ax1.set_ylabel("mAP50 on DamCrack test")
        ax1.set_title("(a) Three arms: annotation size vs. accuracy", fontsize=11)
        ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
        ax1.grid(alpha=0.25)
        ax1.set_xlim(55, 9000)
        # 标注档位名
        # 全部改为点在右下/左下错开，均在图内
        for i, r in enumerate(rows):
            if i == 0:
                xy, ha = (8, -6), "left"
            elif i == 1:
                xy, ha = (8, 4), "left"
            else:
                xy, ha = (0, -15), "center"
            ax1.annotate(r["tier"], (r["gt_boxes"], r["RpT_mean"]),
                         textcoords="offset points", xytext=xy,
                         ha=ha, fontsize=7.5, color="#444444")

        # ---------- (b) 净增益 ----------
        ys = [r["net_mean"] for r in rows]
        errs = [r["net_sd"] for r in rows]
        colors = ["#d62728" if y < 0 else "#2ca02c" for y in ys]
        bars = ax2.bar(range(len(rows)), ys, yerr=errs, capsize=4,
                       color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
        ax2.axhline(0, color="#333333", lw=1.2)

        # 临界区间阴影（shot20 与 shot30 之间）
        ax2.axvspan(1.5, 2.5, color="#ff9900", alpha=0.13, zorder=0)
        ymax_ax = max(y + e for y, e in zip(ys, errs))
        ymin_ax = min(y - e for y, e in zip(ys, errs))
        ax2.set_ylim(ymin_ax - 0.012, ymax_ax + 0.022)
        ax2.text(2.0, ymax_ax + 0.016, "critical\n80–157 boxes",
                 ha="center", va="top", fontsize=8, color="#b36b00", fontweight="bold")

        for i, r in enumerate(rows):
            if r["net_mean"] >= 0:
                y = r["net_mean"] + r["net_sd"] + 0.004
            else:
                y = r["net_mean"] - r["net_sd"] - 0.006
            ax2.text(i, y, f"{r['net_mean']:+.4f}",
                     ha="center", va="bottom" if r["net_mean"] >= 0 else "top",
                     fontsize=9, fontweight="bold")
        ax2.set_xticks(range(len(rows)))
        ax2.set_xticklabels([r["tier"] for r in rows], rotation=18, fontsize=8.5)
        ax2.set_ylabel("Net benefit  (A-PL − repeat-A)")
        ax2.set_title("(b) Net pseudo-label benefit by tier", fontsize=11)
        ax2.grid(alpha=0.25, axis="y")

        # 右下角注明统计口径
        fig.text(0.5, 0.005,
                 "Mean ± population SD (ddof=0) over 3 annotation-sampling seeds (42/43/44); training seed fixed at 42.",
                 ha="center", fontsize=7.6, color="#555555")

        fig.tight_layout(rect=(0, 0.05, 1, 1))
        out_png = RESULTS / "fig4_pseudo_label_3arms.png"
        fig.savefig(out_png, dpi=170)
        print(f"[write] {out_png}")
    except Exception as exc:
        print(f"[warn] 图未生成: {exc}")

    print("\n=== 图 4 数据 ===")
    print(f"{'tier':<8}{'gt':>6}{'A':>9}{'A-PL':>9}{'repeatA':>10}{'net':>10}")
    for r in rows:
        print(f"{r['tier']:<8}{r['gt_boxes']:>6}{r['A_mean']:>9.4f}"
              f"{r['APL_mean']:>9.4f}{r['RpT_mean']:>10.4f}{r['net_mean']:>+10.4f}")


if __name__ == "__main__":
    main()
