# -*- coding: utf-8 -*-
"""30: 方案 A（伪标签自训练）结果汇总 + Δ-标注量曲线。

读取 crossdomain.csv 中：
  - A 组 baseline：tag = <tier>_seed42
  - A-PL（方案 A）：tag = <tier>_seed42_pl   （由 04_eval.py --protocol crossdomain 写入）
计算 Δ = A-PL map50 − A map50，定位"能力启动阈值"。

产物：
  - results/tables/fig_pl_delta.csv   （数据表）
  - results/fig_pl_delta.png          （Δ-标注量曲线，含零线）

运行（须先完成 7 档 A-PL 训练 + 04_eval.py 评估）：
  python scripts/30_pl_report.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from common import RESULTS

# 档位 -> x 轴标注预算（图像数或 instance 数，取 tier 名内数字，日志刻度）
TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
BUDGET = {
    "shot10": 10, "shot20": 20, "shot30": 30, "shot50": 50,
    "img100": 100, "img200": 200, "img500": 500,
}


def main() -> None:
    cross = pd.read_csv(RESULTS / "crossdomain.csv", dtype={"tag": str})
    by_tag = {r["tag"]: r for _, r in cross.iterrows()}

    rows = []
    for tier in TIERS:
        a = by_tag.get(f"{tier}_seed42")
        pl = by_tag.get(f"{tier}_seed42_pl")
        if a is None:
            print(f"[skip] {tier}: 缺 A 组 baseline 行 {tier}_seed42")
            continue
        a50 = float(a["map50"])
        if pl is None:
            print(f"[pending] {tier}: A-PL 尚未评估（缺 {tier}_seed42_pl）")
            rows.append({"tier": tier, "budget": BUDGET[tier], "A_map50": a50,
                         "PL_map50": "", "delta": "", "note": "pending"})
            continue
        pl50 = float(pl["map50"])
        rows.append({"tier": tier, "budget": BUDGET[tier], "A_map50": round(a50, 4),
                     "PL_map50": round(pl50, 4), "delta": round(pl50 - a50, 4),
                     "note": ""})

    dst = RESULTS / "tables"
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "fig_pl_delta.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["tier", "budget", "A_map50", "PL_map50", "delta", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {dst / 'fig_pl_delta.csv'} ({len(rows)} 行)")

    # 只画已评估的档位
    done = [r for r in rows if r["PL_map50"] != ""]
    if not done:
        print("[info] 尚无已评估档位，跳过绘图")
        return

    xs = [r["budget"] for r in done]
    ys = [r["delta"] for r in done]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhline(0, color="#d62728", linestyle="--", linewidth=1, label="Δ = 0 (无增益)")
        ax.plot(xs, ys, "-o", color="#1f77b4", label="A-PL − A (mAP50)")
        # 标注每个点的 Δ 值
        for x, y, r in zip(xs, ys, done):
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(list(BUDGET.values()))
        ax.set_xticklabels([t.replace("shot", "").replace("img", "") for t in TIERS])
        ax.set_xlabel("annotation budget (log scale)")
        ax.set_ylabel("Δ mAP50 (A-PL − A)")
        ax.set_title("Pseudo-label self-training gain vs annotation budget")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_pl_delta.png", dpi=150)
        print("[write] results/fig_pl_delta.png")
    except Exception as exc:
        print(f"[warn] 图未生成: {exc}")


if __name__ == "__main__":
    main()
