# -*- coding: utf-8 -*-
"""35: 方案 A 的 3-seed 汇总（配对 Δ、均值与标准差）。

设计要点（避免统计口径错误）：
  - Δ 必须**同 seed 内配对**计算（A-PL(seed s) − A(seed s)），再对 3 个 seed 求均值。
    不得用不同 seed 的 A 与 A-PL 混配。
  - 标准差用 ddof=0（与 §3.4 表 2 一致，见附录 N03）。
  - 只纳入 3 个 seed 齐全的档位；缺 seed 的档位单独列出，不参与均值。

产物：
  - results/tables/pl_3seed.csv        逐档 3-seed 明细与配对 Δ
  - results/tables/pl_3seed_summary.csv  逐档均值±标准差
  - results/tables/pl_3seed.tex        Springer 三线表

运行：python scripts/35_pl_3seed_summary.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from common import RESULTS

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
SEEDS = (42, 43, 44)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # 95% two-sided, df=n-1


def load() -> dict[str, dict]:
    p = RESULTS / "crossdomain.csv"
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        return {r["tag"].strip(): r for r in csv.DictReader(f)}


def m50(rows: dict, tag: str) -> float | None:
    r = rows.get(tag)
    if not r:
        return None
    v = (r.get("map50") or "").strip()
    return float(v) if v else None


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std_ddof0(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main() -> None:
    rows = load()
    detail: list[dict] = []
    summary: list[dict] = []

    for tier in TIERS:
        a_vals, p_vals, d_vals = [], [], []
        for s in SEEDS:
            a = m50(rows, f"{tier}_seed{s}")
            p = m50(rows, f"{tier}_seed{s}_pl")
            rec = {"tier": tier, "seed": s, "A": "" if a is None else round(a, 4),
                   "A_PL": "" if p is None else round(p, 4),
                   "delta": "" if (a is None or p is None) else round(p - a, 4)}
            detail.append(rec)
            if a is not None and p is not None:
                a_vals.append(a)
                p_vals.append(p)
                d_vals.append(p - a)

        n = len(d_vals)
        if n == 0:
            summary.append({"tier": tier, "n_seeds": 0, "note": "无完整配对"})
            continue
        rec = {
            "tier": tier, "n_seeds": n,
            "A_mean": round(mean(a_vals), 4), "A_std": round(std_ddof0(a_vals), 4),
            "APL_mean": round(mean(p_vals), 4), "APL_std": round(std_ddof0(p_vals), 4),
            "delta_mean": round(mean(d_vals), 4), "delta_std": round(std_ddof0(d_vals), 4),
        }
        if n >= 2:
            sd = std_ddof0(d_vals)
            se = sd / math.sqrt(n)
            tc = T_CRIT.get(n - 1, 2.0)
            rec["delta_ci95_low"] = round(mean(d_vals) - tc * se, 4)
            rec["delta_ci95_high"] = round(mean(d_vals) + tc * se, 4)
            rec["ci_excludes_zero"] = (rec["delta_ci95_low"] > 0) or (rec["delta_ci95_high"] < 0)
        rec["all_positive"] = all(d > 0 for d in d_vals)
        rec["rel_gain_pct"] = round(mean(d_vals) / mean(a_vals) * 100, 1) if mean(a_vals) else ""
        rec["note"] = "" if n == len(SEEDS) else f"仅 {n}/{len(SEEDS)} seed"
        summary.append(rec)

    # ---- CSV ----
    dpath = RESULTS / "tables" / "pl_3seed.csv"
    dpath.parent.mkdir(parents=True, exist_ok=True)
    with open(dpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["tier", "seed", "A", "A_PL", "delta"])
        w.writeheader()
        w.writerows(detail)

    spath = RESULTS / "tables" / "pl_3seed_summary.csv"
    keys = ["tier", "n_seeds", "A_mean", "A_std", "APL_mean", "APL_std",
            "delta_mean", "delta_std", "delta_ci95_low", "delta_ci95_high",
            "ci_excludes_zero", "all_positive", "rel_gain_pct", "note"]
    with open(spath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    print(f"[write] {dpath}")
    print(f"[write] {spath}")

    # ---- 控制台 ----
    print("\n=== 逐档 3-seed 汇总（配对 Δ）===")
    print(f"{'tier':<8}{'n':>3}{'A':>9}{'A-PL':>9}{'Δ':>9}{'Δ±sd':>12}{'95%CI':>18}{'rel%':>8}{'全正':>6}")
    for r in summary:
        if r["n_seeds"] == 0:
            continue
        ci = (f"[{r['delta_ci95_low']:+.3f},{r['delta_ci95_high']:+.3f}]"
              if "delta_ci95_low" in r else "—")
        print(f"{r['tier']:<8}{r['n_seeds']:>3}{r['A_mean']:>9.4f}{r['APL_mean']:>9.4f}"
              f"{r['delta_mean']:>+9.4f}{r['delta_mean']:>+8.4f}±{r['delta_std']:.3f}"
              f"{ci:>18}{r['rel_gain_pct']:>+7.1f}%{'是' if r['all_positive'] else '否':>5}")

    # ---- LaTeX ----
    tex = [
        r"% 方案 A 3-seed 汇总 —— 由 scripts/35_pl_3seed_summary.py 自动生成",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Pseudo-label self-training over three annotation-sampling seeds "
        r"(42/43/44). $\Delta$ is computed \emph{within} each seed "
        r"(A-PL $-$ A) and then averaged, so that seed-specific sampling "
        r"is cancelled. Errors are population standard deviations (ddof $=0$).}",
        r"\label{tab:pseudo-label-3seed}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Tier & $n$ & A (mAP50) & A-PL (mAP50) & $\Delta \pm$ SD & Rel. gain \\",
        r"\midrule",
    ]
    for r in summary:
        if r["n_seeds"] == 0:
            continue
        tex.append(
            f"{r['tier']} & {r['n_seeds']} & {r['A_mean']:.4f}$\\pm${r['A_std']:.4f} & "
            f"{r['APL_mean']:.4f}$\\pm${r['APL_std']:.4f} & "
            f"${r['delta_mean']:+.4f}\\pm{r['delta_std']:.4f}$ & "
            f"{r['rel_gain_pct']:+.1f}\\% \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tpath = RESULTS / "tables" / "pl_3seed.tex"
    tpath.write_text("\n".join(tex), encoding="utf-8")
    print(f"\n[write] {tpath}")


if __name__ == "__main__":
    main()
