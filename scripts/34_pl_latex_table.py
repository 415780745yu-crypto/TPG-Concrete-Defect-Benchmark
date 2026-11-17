# -*- coding: utf-8 -*-
"""34: 生成论文可直接使用的 LaTeX / Markdown 表格（方案 A 结果）。

输入：
  - results/tables/fig_pl_delta_v2.csv   （主数据：gt_boxes / snr / A / PL / delta / rel_gain）
  - results/crossdomain.csv              （分类别 AP）
输出：
  - results/tables/pl_results.tex   Springer sn-jnl 可 \\input 的三线表
  - results/tables/pl_results.md    Markdown 版

运行：python scripts/34_pl_latex_table.py
"""
from __future__ import annotations

import pandas as pd

from common import RESULTS

ORDER = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]


def main() -> None:
    v2 = RESULTS / "tables" / "fig_pl_delta_v2.csv"
    if not v2.exists():
        print(f"[STOP] 缺 {v2}，先跑 32_pl_report_v2.py")
        return
    df = pd.read_csv(v2)
    df["_ord"] = df["tier"].apply(lambda t: ORDER.index(t) if t in ORDER else 99)
    df = df.sort_values("_ord").drop(columns="_ord")

    cross = pd.read_csv(RESULTS / "crossdomain.csv", dtype={"tag": str})

    # 分类别 AP
    def cls_ap(tag: str, col: str) -> float | None:
        hit = cross[cross["tag"] == tag]
        if len(hit) == 0:
            return None
        v = hit[col].iloc[0]
        return None if pd.isna(v) else float(v)

    rows = []
    for r in df.itertuples():
        ac = cls_ap(f"{r.tier}_seed42", "ap50_crack")
        pc = cls_ap(f"{r.tier}_seed42_pl", "ap50_crack")
        asp = cls_ap(f"{r.tier}_seed42", "ap50_spalling")
        psp = cls_ap(f"{r.tier}_seed42_pl", "ap50_spalling")
        rows.append({
            "tier": r.tier,
            "gt_boxes": int(r.gt_boxes),
            "snr": float(r.snr),
            "A": float(r.A_map50),
            "PL": float(r.PL_map50),
            "delta": float(r.delta),
            "rel": float(r.rel_gain_pct),
            "crack_A": ac, "crack_PL": pc,
            "spall_A": asp, "spall_PL": psp,
        })

    # ---- Markdown ----
    md = ["| Tier | GT boxes | pseudo:gt | A mAP50 | A-PL mAP50 | Δ | Rel. gain |",
          "|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['tier']} | {x['gt_boxes']} | 1:{x['snr']:.0f} | {x['A']:.4f} | "
                  f"**{x['PL']:.4f}** | **{x['delta']:+.4f}** | {x['rel']:+.1f}% |")
    (RESULTS / "tables" / "pl_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[write] results/tables/pl_results.md")

    # ---- LaTeX (sn-jnl 三线表) ----
    def f4(v) -> str:
        return "--" if v is None else f"{v:.4f}"

    def f2(v) -> str:
        return "--" if v is None else f"{v:.2f}"

    lines = [
        r"% 方案 A（伪标签自训练）结果表 —— 由 scripts/34_pl_latex_table.py 自动生成",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pseudo-label self-training on the DamCrack target domain. "
        r"All models start from the same source-pretrained checkpoint and are trained "
        r"with identical hyper-parameters (50 epochs, seed 42); the only difference is "
        r"whether pseudo-labelled images are added to the training set. "
        r"GT boxes: number of ground-truth boxes in the annotated subset; "
        r"pseudo:gt: ratio of pseudo-labelled to ground-truth boxes.}",
        r"\label{tab:pseudo-label}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Tier & GT boxes & pseudo:gt & A (mAP50) & A-PL (mAP50) & $\Delta$ & Rel. gain \\",
        r"\midrule",
    ]
    for x in rows:
        lines.append(
            f"{x['tier']} & {x['gt_boxes']} & 1:{x['snr']:.0f} & {x['A']:.4f} & "
            f"{x['PL']:.4f} & ${x['delta']:+.4f}$ & {x['rel']:+.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    (RESULTS / "tables" / "pl_results.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[write] results/tables/pl_results.tex")

    # ---- 分类别 AP 表 ----
    md2 = ["| Tier | crack A→A-PL | spalling A→A-PL |", "|---|---|---|"]
    for x in rows:
        md2.append(f"| {x['tier']} | {f4(x['crack_A'])} → {f4(x['crack_PL'])} | "
                   f"{f4(x['spall_A'])} → {f4(x['spall_PL'])} |")
    (RESULTS / "tables" / "pl_perclass.md").write_text("\n".join(md2) + "\n", encoding="utf-8")
    print("[write] results/tables/pl_perclass.md")

    # ---- 控制台摘要 ----
    print("\n=== 当前 Δ 汇总 ===")
    for x in rows:
        print(f"  {x['tier']:<7} GT={x['gt_boxes']:>5}  1:{x['snr']:>5.0f}  "
              f"A={x['A']:.4f}  PL={x['PL']:.4f}  Δ={x['delta']:+.4f}  ({x['rel']:+.1f}%)")


if __name__ == "__main__":
    main()
