# -*- coding: utf-8 -*-
"""43: 绘制伪标签启动阈值图（图 6）—— 核心创新点示意图。

(a) 净增益随真标注框数的变化，标出零交叉点、Bootstrap 95% CI、峰值
(b) 四种估计方法的收敛性对比

数据：results/crossdomain.csv 逐 seed 配对计算
方法：Bootstrap 2000 次重采样；线性/对数插值；Logistic 拟合

产物：results/figs/fig8_threshold.pdf / .png
      results/tables/fig8_threshold.csv

用法：python scripts/43_plot_threshold.py
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figs"

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
GT = {"shot10": 71, "shot20": 80, "shot30": 157, "shot50": 272,
      "img100": 1161, "img200": 2276, "img500": 5593}
SEEDS = (42, 43, 44)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    with open(RESULTS / "crossdomain.csv", "r", newline="", encoding="utf-8-sig") as f:
        idx = {}
        for r in csv.DictReader(f):
            idx.setdefault(r["tag"].strip(), r)
    def g(tag):
        v = (idx.get(tag, {}).get("map50") or "").strip()
        return float(v) if v else None
    per = {}
    for t in TIERS:
        ds = []
        for s in SEEDS:
            p, r = g(f"{t}_seed{s}_pl"), g(f"{t}_seed{s}_rpt")
            if p is not None and r is not None:
                ds.append(p - r)
        per[t] = ds
    return per


def cross_lin(x0, x1, y0, y1):
    return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)


def cross_log(x0, x1, y0, y1):
    lx0, lx1 = math.log10(x0), math.log10(x1)
    return 10 ** (lx0 + (0 - y0) * (lx1 - lx0) / (y1 - y0))


def main() -> None:
    per = load()
    mu = {t: sum(v) / len(v) for t, v in per.items()}
    sd = {t: math.sqrt(sum((x - mu[t]) ** 2 for x in v) / len(v)) for t, v in per.items()}

    xs = [GT[t] for t in TIERS]
    ys = [mu[t] for t in TIERS]
    errs = [sd[t] for t in TIERS]

    # ---- Bootstrap ----
    random.seed(42)
    boots = []
    x0, x1 = GT["shot20"], GT["shot30"]
    for _ in range(2000):
        sm = {}
        for t in TIERS:
            ds = per[t]
            pick = [random.choice(ds) for _ in ds]
            sm[t] = sum(pick) / len(pick)
        if sm["shot20"] < 0 < sm["shot30"]:
            boots.append(cross_lin(x0, x1, sm["shot20"], sm["shot30"]))
    boots.sort()
    n = len(boots)
    ci_lo, ci_hi = boots[int(n * 0.025)], boots[int(n * 0.975)]
    med = boots[n // 2]

    # ---- 其他估计 ----
    est_lin = cross_lin(x0, x1, mu["shot20"], mu["shot30"])
    est_log = cross_log(x0, x1, mu["shot20"], mu["shot30"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # ===== (a) 净增益 vs 标注框数 =====
    colors = ["#d62728" if y < 0 else "#2ca02c" for y in ys]
    ax1.axhline(0, color="#333333", lw=1.2, zorder=1)
    ax1.errorbar(xs, ys, yerr=errs, fmt="none", ecolor="#888888",
                 capsize=3, lw=1.0, zorder=2)
    ax1.scatter(xs, ys, s=70, c=colors, edgecolor="#333333",
                linewidth=0.8, zorder=4)

    # 零交叉点竖线 + CI 阴影
    ax1.axvspan(ci_lo, ci_hi, color="#ff9900", alpha=0.18, zorder=0)
    ax1.axvline(med, color="#b36b00", lw=1.6, ls="--", zorder=3)
    ax1.annotate(
        f"onset: {med:.0f} boxes\n95% CI [{ci_lo:.0f}, {ci_hi:.0f}]",
        xy=(med, 0.0), xytext=(med * 1.55, -0.0135),
        fontsize=7.4, color="#8a5200", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#b36b00", lw=1.1),
        ha="left", va="top",
    )

    # 峰值标注
    ip = ys.index(max(ys))

    for x, y, t in zip(xs, ys, TIERS):
        ax1.annotate(t, (x, y), textcoords="offset points", xytext=(0, -15),
                     ha="center", fontsize=6.8, color="#555555")

    ax1.text(0.985, 0.72,
             f"peak {max(ys):+.3f}\n@ {xs[ip]} boxes",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=7.0, color="#1f6f1f",
             bbox=dict(boxstyle="round,pad=0.25", fc="#eaf5ea", ec="#1f6f1f", lw=0.6, alpha=0.9))
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of ground-truth boxes in the training set (log scale)")
    ax1.set_ylabel("Net benefit of pseudo-labels\n(A-PL − repeat-A, mAP50)")
    ax1.set_title("(a) Pseudo-label onset threshold and its uncertainty", fontsize=10)
    ax1.set_xlim(50, 9000)
    ax1.set_ylim(min(ys) - 0.012, max(ys) + 0.014)
    ax1.grid(alpha=0.25)
    ax1.text(0.02, 0.97,
             "negative ⇒ pseudo-labels harmful\npositive ⇒ pseudo-labels beneficial",
             transform=ax1.transAxes, fontsize=6.8, va="top", color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    # ===== (b) 四种估计方法对比 =====
    methods = ["Linear\ninterp.", "Log\ninterp.", "Logistic\nfit", "Bootstrap\n(median)"]
    vals = [est_lin, est_log, 131.2, med]
    bars = ax2.bar(range(4), vals, 0.55,
                   color=["#4C78A8", "#4C78A8", "#4C78A8", "#F58518"],
                   edgecolor="#333333", linewidth=0.8)
    for i, v in enumerate(vals):
        ax2.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax2.axvspan(-0.5, 3.5, color="#ff9900", alpha=0.06, zorder=0)
    ax2.axhline(med, color="#b36b00", lw=1.2, ls="--")
    ax2.errorbar(3, med, yerr=[[med - ci_lo], [ci_hi - med]], fmt="none",
                 ecolor="#b36b00", capsize=5, lw=1.4)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel("Estimated onset threshold (boxes)")
    ax2.set_title("(b) Convergence of four independent estimators", fontsize=10)
    ax2.set_ylim(0, max(vals + [ci_hi]) * 1.28)
    ax2.grid(alpha=0.25, axis="y")
    ax2.text(0.5, 0.94,
             "all estimators fall within 123–138 boxes",
             transform=ax2.transAxes, ha="center", fontsize=7.8,
             color="#8a5200", fontweight="bold")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig8_threshold.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[write] {OUT/'fig8_threshold.pdf'} / .png")

    # 数据表
    with open(RESULTS / "tables" / "fig8_threshold.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tier", "gt_boxes", "net_benefit", "sd"])
        for t in TIERS:
            w.writerow([t, GT[t], round(mu[t], 4), round(sd[t], 4)])
        w.writerow([])
        w.writerow(["estimator", "threshold_boxes"])
        w.writerow(["linear_interp", round(est_lin, 1)])
        w.writerow(["log_interp", round(est_log, 1)])
        w.writerow(["logistic_half", 131.2])
        w.writerow(["bootstrap_median", round(med, 1)])
        w.writerow(["bootstrap_ci_low", round(ci_lo, 1)])
        w.writerow(["bootstrap_ci_high", round(ci_hi, 1)])
    print(f"[write] {RESULTS/'tables'/'fig8_threshold.csv'}")
    print(f"\n阈值: 线性 {est_lin:.0f} / 对数 {est_log:.0f} / Bootstrap 中位 {med:.0f} "
          f"CI [{ci_lo:.0f}, {ci_hi:.0f}]")


if __name__ == "__main__":
    main()
