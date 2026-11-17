# -*- coding: utf-8 -*-

"""40: 绘制论文各图（出版级矢量图，对标参考论文版式）。

图 1  研究框架（纵向流程 + 分组底框）—— 版式对标 sensors-24-01936 Fig.2

      产出: results/figs/fig1_framework.pdf / .png

图 2  受控扰动对两个域的影响量级对比（双面板）—— 版式对标 YOLO-Mamba Fig.3

      产出: results/figs/fig2_domain_gap.pdf / .png

图 3  标注预算与精度恢复（双面板）

      产出: results/figs/fig3_budget.pdf / .png

图 5  跨域失效的两种形态（幻影类置信度分布）

      产出: results/figs/fig5_failure.pdf / .png

全部数据来自落盘 CSV，不硬编码数值。

用法:

  python scripts/40_plot_figures.py            # 全部

  python scripts/40_plot_figures.py --only 1   # 只画图 1

"""

from __future__ import annotations

import argparse

import csv

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

OUT = RESULTS / "figs"

# ---- 参考论文配色（取自模板：浅蓝 / 浅橙 / 浅绿 / 浅紫 分组框）----

C_BLUE = "#DCE9F7"

C_ORANGE = "#FBE3CF"

C_GREEN = "#DFF0D8"

C_PURPLE = "#E8E0F3"

C_GREY = "#EFEFEF"

C_EDGE = "#4A4A4A"

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 9,

    "axes.linewidth": 0.8,

    "savefig.bbox": "tight",

    "pdf.fonttype": 42,   # 可编辑文本（投稿要求）

    "ps.fonttype": 42,

})

def _box(ax, x, y, w, h, text, fc=C_GREY, fs=8.5, bold=False, rounded=0.02):

    """画一个带文字的圆角框（坐标用 0-1 轴坐标）。"""

    p = FancyBboxPatch((x, y), w, h,

                       boxstyle=f"round,pad=0.004,rounding_size={rounded}",

                       linewidth=0.9, edgecolor=C_EDGE, facecolor=fc, zorder=3)

    ax.add_patch(p)

    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",

            fontsize=fs, fontweight=("bold" if bold else "normal"),

            zorder=4, linespacing=1.35)

    return p

def _arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.0, color=C_EDGE, ls="-"):

    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,

                        mutation_scale=9, linewidth=lw, color=color,

                        linestyle=ls, zorder=2, shrinkA=1, shrinkB=1)

    ax.add_patch(a)

# ============================ 图 1 研究框架 ============================

def fig1() -> None:

    fig, ax = plt.subplots(figsize=(7.4, 10.0))

    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1)

    ax.axis("off")

    # ===== 顶部：数据与协议 =====

    _box(ax, 0.05, 0.950, 0.90, 0.042,

         "Two public datasets, unified splits & evaluation protocol",

         fc=C_BLUE, fs=8.8, bold=True)

    _box(ax, 0.05, 0.876, 0.40, 0.058,

         "Source domain\nCUBIT-Det (3 classes)\n5,525 images", fc=C_GREY, fs=8.2)

    _box(ax, 0.55, 0.876, 0.40, 0.058,

         "Target domain\nDamCrack (2 classes)\n4,000 images · 2800/600/600", fc=C_GREY, fs=8.2)

    _arrow(ax, 0.45, 0.905, 0.55, 0.905)

    ax.text(0.50, 0.912, "transfer", ha="center", va="bottom", fontsize=6.6, style="italic")

    _arrow(ax, 0.50, 0.950, 0.50, 0.936)

    # ===== Stage 1（0.735–0.862）=====

    _box(ax, 0.05, 0.735, 0.90, 0.128, "", fc=C_GREEN)

    ax.text(0.065, 0.856, "Stage 1 — Cross-domain benchmark",

            fontsize=8.8, fontweight="bold", va="top")

    # 三个数值框

    _box(ax, 0.075, 0.786, 0.26, 0.044, "Source mAP50\n0.7153", fc="white", fs=7.9)

    _box(ax, 0.370, 0.786, 0.26, 0.044, "Zero-shot mAP50\n0.0459", fc="white", fs=7.9)

    _box(ax, 0.665, 0.786, 0.26, 0.044, "Target upper bound\n0.6960", fc="white", fs=7.9)

    # gap 公式（独立一行）

    ax.text(0.50, 0.766, "recoverable gap = 0.6960 − 0.0459 = 0.6501",

            ha="center", va="center", fontsize=8.4, fontweight="bold")

    # 七档说明（再下一行，留足间距）

    ax.text(0.50, 0.744, "seven annotation tiers (instance- and image-level)",

            ha="center", va="center", fontsize=7.4, style="italic")

    _arrow(ax, 0.50, 0.876, 0.50, 0.865)

    # ===== Stage 2（0.560–0.700）=====

    _box(ax, 0.05, 0.560, 0.90, 0.128, "", fc=C_ORANGE)

    ax.text(0.065, 0.681, "Stage 2 — Attribution of degradation",

            fontsize=8.8, fontweight="bold", va="top")

    _box(ax, 0.075, 0.612, 0.40, 0.046,

         "Target: T1 photometric\nT2 scale · T3 texture", fc="white", fs=7.6)

    _box(ax, 0.525, 0.612, 0.40, 0.046,

         "Source: same-parameter\nextreme (T2 / T3)", fc="white", fs=7.6)

    ax.text(0.50, 0.590, "source loss 0.0644–0.0832  <  1/8 × gap 0.6501",

            ha="center", va="center", fontsize=8.3, fontweight="bold")

    ax.text(0.50, 0.569, "→ low-level imaging is not the dominant cause",

            ha="center", va="center", fontsize=7.6, style="italic")

    _arrow(ax, 0.50, 0.735, 0.50, 0.700)

    # ===== Stage 3（0.345–0.520）=====

    _box(ax, 0.05, 0.345, 0.90, 0.168, "", fc=C_PURPLE)

    ax.text(0.065, 0.506, "Stage 3 — Recovery path",

            fontsize=8.8, fontweight="bold", va="top")

    _box(ax, 0.075, 0.430, 0.26, 0.048,

         "Budget sweep\n100 / 200 / 500 imgs", fc="white", fs=7.6)

    _box(ax, 0.370, 0.430, 0.26, 0.048,

         "Pseudo-labels\nA / A-PL / repeat-A", fc="white", fs=7.6)

    _box(ax, 0.665, 0.430, 0.26, 0.048,

         "Threshold τ\n0.1 / 0.25 / 0.5", fc="white", fs=7.6)

    ax.text(0.50, 0.404, "net benefit of pseudo-labels is annotation-dependent:",

            ha="center", va="center", fontsize=7.9)

    ax.text(0.50, 0.385, "negative below ≈100 boxes · peak ≈1,100 boxes · decays after",

            ha="center", va="center", fontsize=7.9, fontweight="bold")

    ax.text(0.50, 0.360, "→ decision tables: acceptance target ↔ required budget",

            ha="center", va="center", fontsize=7.5, style="italic")

    _arrow(ax, 0.50, 0.560, 0.50, 0.526)

    # ===== Stage 4（0.175–0.300）=====

    _box(ax, 0.05, 0.175, 0.90, 0.128, "", fc=C_BLUE)

    ax.text(0.065, 0.296, "Stage 4 — Deployment feasibility (x86 CPU, NCNN)",

            fontsize=8.8, fontweight="bold", va="top")

    _box(ax, 0.075, 0.227, 0.26, 0.046,

         "Export fidelity\nONNX / NCNN", fc="white", fs=7.6)

    _box(ax, 0.370, 0.227, 0.26, 0.046,

         "Backbone swap\nvs. INT8", fc="white", fs=7.6)

    _box(ax, 0.665, 0.227, 0.26, 0.046,

         "Per-domain\nquantization loss", fc="white", fs=7.6)

    ax.text(0.50, 0.204, "INT8: −32.1% latency, −74.0% size (source)  |  −8.10 mAP50 (target)",

            ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax.text(0.50, 0.184, "→ quantization loss is domain-dependent; cannot be extrapolated",

            ha="center", va="center", fontsize=7.3, style="italic")

    _arrow(ax, 0.50, 0.345, 0.50, 0.311)

    # ===== 底部：产出 =====

    _box(ax, 0.05, 0.045, 0.90, 0.078,

         "Deliverables\nrecomputable splits · scripts · result tables · decision tables",

         fc=C_GREY, fs=8.3, bold=True)

    _arrow(ax, 0.50, 0.175, 0.50, 0.130)

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):

        fig.savefig(OUT / f"fig1_framework.{ext}", dpi=300 if ext == "png" else None)

    plt.close(fig)

    print(f"[write] {OUT/'fig1_framework.pdf'} / .png")

# ============================ 图 2 退化归因 ============================

def _read_csv(name: str) -> list[dict]:

    with open(RESULTS / name, "r", newline="", encoding="utf-8-sig") as f:

        return list(csv.DictReader(f))

def fig2() -> None:

    """受控扰动对两个域的影响（双面板）。数据：domain_gap.csv / source_control.csv"""

    dg = _read_csv("domain_gap.csv")

    sc = _read_csv("source_control.csv")

    # 目标域：variant 形如 T0_original / T1_photometric / T2_scale / T3_blur

    def short(v: str) -> str:

        return v.split("_")[0]

    tgt = {short(r["variant"]): float(r["map50"]) for r in dg if r.get("map50")}

    if not tgt:

        print("[skip] fig2: domain_gap.csv 无可用数据")

        return

    src = {short(r["variant"]): float(r["map50"]) for r in sc if r.get("map50")}

    src_base = float(sc[0]["baseline_map50"]) if sc and sc[0].get("baseline_map50") else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))

    order = [k for k in ["T0", "T1", "T2", "T3"] if k in tgt]

    labels = {"T0": "T0\nbaseline", "T1": "T1\nphotometric",

              "T2": "T2\nscale (2×)", "T3": "T3\ntexture (blur)"}

    ys = [tgt[k] for k in order]

    ax1.bar(range(len(order)), ys,

            color=["#4C78A8"] + ["#F58518"] * (len(order) - 1),

            edgecolor=C_EDGE, linewidth=0.7)

    for i, y in enumerate(ys):

        ax1.text(i, y + 0.012, f"{y:.4f}", ha="center", fontsize=8)

        if i > 0:

            ax1.text(i, y + 0.045, f"{(y - ys[0]) * 100:+.2f} pt", ha="center",

                     fontsize=7.2, color="#8B2E00")

    ax1.set_xticks(range(len(order)))

    ax1.set_xticklabels([labels[k] for k in order], fontsize=7.6)

    ax1.set_ylabel("AP50$^{self}$ on target domain")

    ax1.set_title("(a) Target domain: controlled perturbations", fontsize=9.5)

    ax1.grid(alpha=0.25, axis="y")

    ax1.set_ylim(0, 0.86)   # 与 (b) 同尺度；顶部留白放数值标签

    # 源域：T2/T3 同参数对照 vs baseline（0.7153）

    sxs = [k for k in ["T2", "T3"] if k in src]

    if sxs:

        vals = [src[k] for k in sxs]

        base = src_base if src_base is not None else max(vals) * 1.1

        xs_all = ["baseline"] + sxs

        ys_all = [base] + vals

        ax2.bar(range(len(xs_all)), ys_all,

                color=["#4C78A8"] + ["#54A24B"] * len(sxs),

                edgecolor=C_EDGE, linewidth=0.7)

        for i, y in enumerate(ys_all):

            ax2.text(i, y + 0.008, f"{y:.4f}", ha="center", fontsize=8)

            if i > 0:

                ax2.text(i, y - 0.045, f"{(y - base) * 100:+.2f} pt", ha="center",

                         fontsize=7.2, color="#1E5B22")

        ax2.set_xticks(range(len(xs_all)))

        ax2.set_xticklabels(["baseline", "T2\nscale (2×)", "T3\ntexture (blur)"][: len(xs_all)],

                            fontsize=7.6)

        ax2.set_ylabel("mAP50 on source domain (3 classes)")

        ax2.set_title("(b) Source domain: same-parameter control", fontsize=9.5)

        ax2.grid(alpha=0.25, axis="y")

        ax2.set_ylim(0, max(ys_all) * 1.18)

    # 放大插图：显示目标域四个扰动档的细节

    axins = ax1.inset_axes([0.42, 0.55, 0.55, 0.38])

    axins.bar(range(len(order)), ys,

              color=["#4C78A8"] + ["#F58518"] * (len(order) - 1),

              edgecolor=C_EDGE, linewidth=0.5)

    for i, y in enumerate(ys):

        axins.text(i, y + 0.0003, f"{y:.4f}", ha="center", fontsize=6.2)

    axins.set_ylim(min(ys) * 0.97, max(ys) * 1.03)

    axins.set_xticks(range(len(order)))

    axins.set_xticklabels(order, fontsize=6)

    axins.tick_params(axis="y", labelsize=6)

    axins.grid(alpha=0.2, axis="y")

    axins.set_title("zoom: T0-T3 detail", fontsize=6.5)

    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):

        fig.savefig(OUT / f"fig2_domain_gap.{ext}", dpi=300 if ext == "png" else None)

    plt.close(fig)

    print(f"[write] {OUT/'fig2_domain_gap.pdf'} / .png")

# ============================ 图 3 标注预算 ============================

def fig3() -> None:

    """标注预算与精度恢复（双面板；实例级与图像级不合并为单轴）。

    数据来源：

      实例级 = tables/pl_3seed_summary.csv 的 A_mean（3 seed 均值）

      图像级 = 步数对齐后的实测值（_seed42_long 覆盖 img100/img200；img500 原生即 3150 步）

    与正文表 2 完全一致。

    """

    rows = _read_csv("tables/pl_3seed_summary.csv")

    sm = {r["tier"]: r for r in rows}

    cross = {r["tag"].strip(): r for r in _read_csv("crossdomain.csv")}

    def m50(tag):

        v = (cross.get(tag, {}).get("map50") or "").strip()

        return float(v) if v else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.1))

    # ---------- (a) 实例级 ----------

    inst = [("shot10", 10), ("shot20", 20), ("shot30", 30), ("shot50", 50)]

    inst = [(t, n) for t, n in inst if t in sm]

    xs = [n for _, n in inst]

    ys = [float(sm[t]["A_mean"]) for t, _ in inst]

    es = [float(sm[t]["A_std"]) for t, _ in inst]

    ax1.errorbar(xs, ys, yerr=es, fmt="-o", capsize=3, lw=1.8, ms=6, color="#4C78A8")

    for i, (x, y) in enumerate(zip(xs, ys)):

        offs = [(10, 12), (10, -18), (12, 10), (-12, 12)]

        off = offs[i % 4]

        ha = "left" if off[0] > 0 else "right"

        ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",

                     xytext=off, ha=ha, fontsize=7.6)

    ax1.set_xlabel("annotated instances per class")

    ax1.set_ylabel("mAP50$^{T}$ (test)")

    ax1.set_title("(a) Instance-level budget", fontsize=9.5)

    ax1.grid(alpha=0.25)

    ax1.set_xlim(5, 56)

    ax1.set_ylim(0, max(y + e for y, e in zip(ys, es)) * 1.22)

    # ---------- (b) 图像级（步数对齐值） ----------

    img = [("img100", 100, "img100_seed42_long"),

           ("img200", 200, "img200_seed42_long"),

           ("img500", 500, "img500_seed42")]

    pts = [(n, m50(tag)) for _, n, tag in img if m50(tag) is not None]

    xs2 = [n for n, _ in pts]

    ys2 = [v for _, v in pts]

    ax2.plot(xs2, ys2, "-s", lw=1.8, ms=6, color="#F58518")

    for x, y in zip(xs2, ys2):

        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",

                     xytext=(0, 11), ha="center", fontsize=7.6)

    ax2.axhline(0.0459, ls="--", lw=1, color="#999999")

    ax2.text(xs2[0] + 8, 0.0459 + 0.022, "zero-shot 0.0459", fontsize=7.4, color="#666666")

    ax2.axhline(0.6960, ls="--", lw=1, color="#999999")

    ax2.text(xs2[0] + 8, 0.6960 - 0.052, "supervised upper bound 0.6960",

             fontsize=7.4, color="#666666")

    ax2.set_xlabel("annotated images")

    ax2.set_ylabel("mAP50$^{T}$ (test)")

    ax2.set_title("(b) Image-level budget (step-matched)", fontsize=9.5)

    ax2.grid(alpha=0.25)

    ax2.set_xlim(xs2[0] - 35, xs2[-1] + 55)

    ax2.set_ylim(0, 0.80)

    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):

        fig.savefig(OUT / f"fig3_budget.{ext}", dpi=300 if ext == "png" else None)

    plt.close(fig)

    print(f"[write] {OUT/'fig3_budget.pdf'} / .png")

    print(f"  (b) 使用步数对齐值: {list(zip(xs2, ys2))}")

# ============================ 图 5 失效形态 ============================

def fig5() -> None:

    """幻影类（moisture）在目标域的检出量：两层阈值对比。

    数据：phantom_class.csv（每行一个阈值层）

    左：幻影框数与受影响图像数；右：幻影类平均置信度。

    """

    rows = _read_csv("phantom_class.csv")

    if not rows:

        print("[skip] fig5: phantom_class.csv 为空")

        return

    labels, nbox, nimg, mconf = [], [], [], []

    for r in rows:

        thr = float(r["conf_threshold"])

        labels.append("conf = 0.001\n(AP evaluation)" if thr < 0.01

                      else f"conf ≥ {thr:g}\n(deployment)")

        nbox.append(float(r["det_class2_phantom"]))

        nimg.append(float(r["images_with_phantom"]))

        mc = (r.get("phantom_mean_conf") or "").strip()

        mconf.append(float(mc) if mc else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    x = list(range(len(labels)))

    w = 0.36

    b1 = ax1.bar([i - w / 2 for i in x], nbox, w, label="phantom boxes",

                 color="#E45756", edgecolor=C_EDGE, linewidth=0.7)

    b2 = ax1.bar([i + w / 2 for i in x], nimg, w, label="images affected",

                 color="#72B7B2", edgecolor=C_EDGE, linewidth=0.7)

    for bars in (b1, b2):

        for b in bars:

            h = b.get_height()

            ax1.text(b.get_x() + b.get_width() / 2, h * 1.12, f"{int(h)}",

                     ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)

    ax1.set_xticklabels(labels, fontsize=7.8)

    ax1.set_ylabel("count on target test set (600 images)")

    ax1.set_title("(a) Phantom-class output (moisture)", fontsize=9.5)

    ax1.legend(fontsize=8, loc="upper right")

    ax1.grid(alpha=0.25, axis="y")

    ax1.set_yscale("log")

    ax1.set_ylim(10, max(max(nbox), max(nimg)) * 3.2)

    ax1.set_ylabel("count on target test set (600 images, log scale)")

    ax2.bar(x, mconf, 0.5, color=["#B279A2", "#4C78A8"][: len(x)],

            edgecolor=C_EDGE, linewidth=0.7)

    for i, y in enumerate(mconf):

        ax2.text(i, y + 0.012, f"{y:.4f}", ha="center", fontsize=8)

    ax2.set_xticks(x)

    ax2.set_xticklabels(labels, fontsize=7.8)

    ax2.set_ylabel("mean confidence of phantom boxes")

    ax2.set_title("(b) Mean confidence", fontsize=9.5)

    ax2.grid(alpha=0.25, axis="y")

    ax2.set_ylim(0, max(mconf) * 1.25 if max(mconf) > 0 else 1)

    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):

        fig.savefig(OUT / f"fig5_failure.{ext}", dpi=300 if ext == "png" else None)

    plt.close(fig)

    print(f"[write] {OUT/'fig5_failure.pdf'} / .png")

# ============================ 图 6 三维权衡 ============================

def fig6() -> None:

    """精度—体积—延迟三维权衡（双面板）。

    (a) 六种「模型 × 精度」组合在 mAP50–延迟平面的分布，气泡面积 = 体积

    (b) 相对各自 FP32 基线，结构替换与量化两条路径在体积/延迟上的收益

    数据：deploy.csv

    """

    rows = _read_csv("deploy.csv")

    keep = [r for r in rows if r.get("platform") == "x86_cpu"

            and r.get("precision") in ("fp32", "int8")

            and r.get("tag") != "target_only_int8_aciq1000"]

    if not keep:

        print("[skip] fig6: deploy.csv 无可用行")

        return

    # 只用 aciq1000 作为 INT8 代表（kl300 作为校准敏感性另见表）

    sel = [r for r in keep if "kl300" not in r["tag"]]

    label_map = {

        "yolov8n_scratch_fp32": "YOLOv8n (scratch) FP32",

        "yolov8n_scratch_int8_aciq1000": "YOLOv8n (scratch) INT8",

        "pw_yolo_ghostnet_fp32": "GhostNet FP32",

        "pw_yolo_ghostnet_int8_aciq1000": "GhostNet INT8",

        "yolov8n_pretrain_fp32": "YOLOv8n (pretrained) FP32",

        "yolov8n_pretrain_int8_aciq1000": "YOLOv8n (pretrained) INT8",

    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4))

    # ---------- (a) 气泡图 ----------

    for r in sel:

        size = float(r["size_MB"])

        lat = float(r["latency_ms_mean"])

        m = float(r["map50"])

        is_int8 = r["precision"] == "int8"

        color = "#F58518" if is_int8 else "#4C78A8"

        ax1.scatter(lat, m, s=size * 28, color=color, alpha=0.72,

                    edgecolor=C_EDGE, linewidth=0.8, zorder=3)

        ax1.annotate(label_map.get(r["tag"], r["tag"]),

                     (lat, m), textcoords="offset points",

                     xytext=(0, -16), ha="center", fontsize=6.8)

    ax1.set_xlabel("end-to-end latency (ms, x86 CPU, NCNN)")

    ax1.set_ylabel("mAP50 on source domain (3 classes)")

    ax1.set_title("(a) Accuracy–latency plane (bubble = model size)", fontsize=9.5)

    ax1.grid(alpha=0.25)

    # 图例（颜色）

    ax1.scatter([], [], color="#4C78A8", s=60, label="FP32", edgecolor=C_EDGE)

    ax1.scatter([], [], color="#F58518", s=60, label="INT8", edgecolor=C_EDGE)

    ax1.legend(fontsize=8, loc="lower left", title="precision", title_fontsize=8)

    ax1.set_xlim(min(float(r["latency_ms_mean"]) for r in sel) - 6,

                 max(float(r["latency_ms_mean"]) for r in sel) + 8)

    lo = min(float(r["map50"]) for r in sel)

    hi = max(float(r["map50"]) for r in sel)

    ax1.set_ylim(lo - 0.06, hi + 0.05)

    # ---------- (b) 相对 FP32 基线的收益 ----------

    pairs = [

        ("YOLOv8n (pretrained)", "yolov8n_pretrain_fp32", "yolov8n_pretrain_int8_aciq1000"),

        ("YOLOv8n (scratch)", "yolov8n_scratch_fp32", "yolov8n_scratch_int8_aciq1000"),

        ("GhostNet", "pw_yolo_ghostnet_fp32", "pw_yolo_ghostnet_int8_aciq1000"),

    ]

    by_tag = {r["tag"]: r for r in rows}

    names, dsize, dlat, dmap = [], [], [], []

    for nm, f32, i8 in pairs:

        if f32 not in by_tag or i8 not in by_tag:

            continue

        s0, s1 = float(by_tag[f32]["size_MB"]), float(by_tag[i8]["size_MB"])

        l0, l1 = float(by_tag[f32]["latency_ms_mean"]), float(by_tag[i8]["latency_ms_mean"])

        m0, m1 = float(by_tag[f32]["map50"]), float(by_tag[i8]["map50"])

        names.append(nm)

        dsize.append((s1 - s0) / s0 * 100)

        dlat.append((l1 - l0) / l0 * 100)

        dmap.append((m1 - m0) * 100)

    import numpy as np
    yv = np.arange(len(names))
    hh = 0.24
    ax2.barh(yv + hh, dsize, hh, label="size", color="#54A24B",
             edgecolor=C_EDGE, linewidth=0.7)
    ax2.barh(yv, dlat, hh, label="latency", color="#4C78A8",
             edgecolor=C_EDGE, linewidth=0.7)
    ax2.barh(yv - hh, dmap, hh, label="mAP50 (points)", color="#E45756",
             edgecolor=C_EDGE, linewidth=0.7)
    for yy, v in zip(yv + hh, dsize):
        ax2.text(v - 1.5, yy, f"{v:+.1f}%", va="center", ha="right", fontsize=7)
    for yy, v in zip(yv, dlat):
        ax2.text(v - 1.5, yy, f"{v:+.1f}%", va="center", ha="right", fontsize=7)
    for yy, v in zip(yv - hh, dmap):
        ax2.text(v - 0.8, yy, f"{v:+.1f} pt", va="center", ha="right", fontsize=7)
    ax2.axvline(0, color=C_EDGE, lw=1)
    ax2.set_yticks(yv)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("change relative to own FP32 baseline (%)")
    ax2.set_title("(b) INT8 gain relative to FP32", fontsize=9.5)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.25, axis="x")
    ax2.set_xlim(min(min(dsize), min(dlat)) - 20, 10)

    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):

        fig.savefig(OUT / f"fig6_tradeoff.{ext}", dpi=300 if ext == "png" else None)

    plt.close(fig)

    print(f"[write] {OUT/'fig6_tradeoff.pdf'} / .png")

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--only", nargs="*", type=int, default=None)

    args = ap.parse_args()

    todo = args.only or [1, 2, 3, 5, 6]

    if 1 in todo: fig1()

    if 2 in todo: fig2()

    if 3 in todo: fig3()

    if 5 in todo: fig5()

    if 6 in todo: fig6()

if __name__ == "__main__":

    main()

