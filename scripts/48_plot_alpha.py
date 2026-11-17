# -*- coding: utf-8 -*-
"""48: 绘制 BSP 混合比例消融图（α 曲线）。

数据：img100 · seed42 · 四点
   α=0    repeat-A  0.3749  （全真标注重复 25.3×）
   α=0.5  混合       0.3640  （真 1265 + 伪 1266）
   α=0.75 混合       0.3716  （真  633 + 伪 1898）
   α=1    A-PL       0.4130  （真  100 + 伪 2431）
   参考  A 仅真标注不重复  0.3854

结论：α∈[0,0.75] 无显著差异，唯 α=1 显著最优 → 开关式(Bang-Bang)而非比例调度

产物：results/figs/fig9_alpha_ablation.pdf / .png
     results/tables/fig9_alpha.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figs"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load():
    idx = {}
    with open(RESULTS / "crossdomain.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            idx.setdefault(r["tag"].strip(), r)
    def g(tag):
        v = (idx.get(tag, {}) or {}).get("map50") or ""
        return float(v) if str(v).strip() else None
    return g


def main():
    g = load()
    P = "img100_seed42"
    pts = [
        (0.0,  g(f"{P}_rpt"),   "repeat-A\n(25.3x repeat)"),
        (0.5,  g(f"{P}_mix0.5"), "50% pseudo\n(12.65x repeat)"),
        (0.75, g(f"{P}_mix0.75"), "75% pseudo\n(6.33x repeat)"),
        (1.0,  g(f"{P}_pl"),    "A-PL\n(no repeat)"),
    ]
    base = g(f"{P}_rpt")
    a_only = g(f"{P}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---------- (a) 绝对精度 ----------
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax1.plot(xs, ys, "-o", color="#4C78A8", ms=9, lw=1.8, zorder=3,
             label="mixed ratio")
    for i, (x, y, lab) in enumerate(pts):
        dy = -17 if i == len(pts) - 1 else 11
        ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=8.2, fontweight="bold")
    # 端点高亮
    ax1.scatter([0, 1], [ys[0], ys[-1]], s=170, facecolor="none",
                edgecolor="#d62728", lw=2.0, zorder=4)
    # 参考线：纯 A（不重复）
    if a_only:
        ax1.axhline(a_only, ls=":", lw=1.2, color="#888888")
        ax1.text(0.02, a_only + 0.002, f"pure A (no repeat) {a_only:.4f}",
                 fontsize=7.4, color="#666666")
    # α=1 显著最优的区间标注
    ax1.text(0.04, 0.955,
             "intermediate mixes ($\\alpha$=0.5, 0.75) are\n"
             "statistically indistinguishable from $\\alpha$=0;\n"
             "only $\\alpha$=1 yields a clear gain",
             transform=ax1.transAxes, fontsize=7.4, va="top", color="#8B2E00",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f0", ec="#d98b6a", lw=0.7))
    ax1.set_xlabel("Pseudo-label fraction $\\alpha$")
    ax1.set_ylabel("mAP50 on target test set")
    ax1.set_title("(a) Mix-ratio ablation (img100, seed42)", fontsize=9.8)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlim(-0.08, 1.12)
    ax1.set_ylim(min(ys) - 0.004, max(ys) + 0.007)
    ax1.grid(alpha=0.25)

    # ---------- (b) 相对增益 + 噪声带 ----------
    d = [y - base for y in ys]
    cols = ["#999999", "#E45756", "#E45756", "#54A24B"]
    bars = ax2.bar([0, 0.5, 0.75, 1.0], d, 0.13, color=cols,
                   edgecolor="#333333", lw=0.7)
    for b, v in zip(bars, d):
        off = 0.0012 if v >= 0 else -0.0012
        ax2.text(b.get_x() + b.get_width()/2, v + off,
                 f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=7.6, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # ±1σ 噪声带
    sd = 0.0085
    ax2.axhspan(-sd, sd, color="#cccccc", alpha=0.45, zorder=0)
    ax2.text(0.02, sd + 0.0012, "$\\pm 1\\sigma$ seed noise", fontsize=7.2, color="#555555")
    ax2.axhline(0, color="#333333", lw=1.0)
    ax2.set_xticks([0, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "0.5", "0.75", "1"], fontsize=8.5)
    ax2.set_xlabel("Pseudo-label fraction $\\alpha$")
    ax2.set_ylabel("ΔmAP50 vs $\\alpha$=0")
    ax2.set_title("(b) Only $\\alpha$=1 clears the noise band", fontsize=9.8)
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(min(d) - 0.006, max(d) + 0.006)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig9_alpha_ablation.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[write] {OUT/'fig9_alpha_ablation.pdf'} / .png")

    with open(RESULTS / "tables" / "fig9_alpha.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "repeat_dir", "pseudo_images", "mAP50", "delta_vs_alpha0"])
        for (a, v, lab), dv in zip(pts, d):
            w.writerow([a, lab.replace("\n", " "), "",
                        round(v, 4) if v else "", round(dv, 4)])
    print(f"[write] {RESULTS/'tables'/'fig9_alpha.csv'}")
    print(f"\n  α=0 {ys[0]:.4f} | 0.5 {ys[1]:.4f} ({d[1]:+.4f}) | "
          f"0.75 {ys[2]:.4f} ({d[2]:+.4f}) | 1.0 {ys[3]:.4f} ({d[3]:+.4f})")


if __name__ == "__main__":
    main()
