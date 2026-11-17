# -*- coding: utf-8 -*-
"""47: BSP 试点结果判定 —— α=0.5 是否优于两端点。

判定规则（seed42 配对，不使用 3 种子均值）：
  img100·seed42 基线：
    α=0   repeat-A = 0.3749
    α≈1  A-PL     = 0.4130   （差 +0.0381）

  ρ = (α=0.5 结果 − α=0) / (α=1 − α=0)

  ρ > 1.0         → 内部最优，BSP 成立（连续调度有意义）
  0 < ρ ≤ 1.0     → 单调递增，α=1 已近优 → 退回 TPG 二值门控
  ρ ≤ 0           → α=0.5 反而差于全真标注 → 强烈支持 TPG 门控

用法：python scripts/47_bsp_verdict.py [--alpha 0.5]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    idx = {}
    with open(ROOT / "results" / "crossdomain.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            idx.setdefault(r["tag"].strip(), r)
    def g(tag):
        v = (idx.get(tag, {}) or {}).get("map50") or ""
        return float(v) if str(v).strip() else None
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tier", default="img100")
    ap.add_argument("--seed", default="seed42")
    args = ap.parse_args()

    g = load()
    a0 = g(f"{args.tier}_{args.seed}_rpt")          # 全真标注（重复至等量）
    a1 = g(f"{args.tier}_{args.seed}_pl")           # 全伪标签
    am = g(f"{args.tier}_{args.seed}_mix{args.alpha:g}")
    a_only = g(f"{args.tier}_{args.seed}")          # 纯 A（不重复）

    print("=" * 66)
    print(f"BSP 判定 · {args.tier} · {args.seed} · α={args.alpha:g}")
    print("=" * 66)
    print(f"\n  α=0    repeat-A（全真标注重复）: {a0 if a0 else '缺失'}")
    print(f"  α={args.alpha:g}  混合比例          : {am if am else '缺失/未完成'}")
    print(f"  α≈1    A-PL（全伪标签）        : {a1 if a1 else '缺失'}")
    if a_only:
        print(f"  （参考）A 仅真标注不重复       : {a_only}")

    if am is None or a0 is None or a1 is None:
        print("\n  ⏳ 数据未齐，无法判定")
        return

    span = a1 - a0
    rho = (am - a0) / span if span else float("nan")

    print(f"\n  端点跨度（α=1 − α=0）: {span:+.4f}")
    print(f"  混合点相对增益 ρ      : {rho:.3f}")

    print("\n" + "-" * 66)
    if rho > 1.0:
        print("  ✅ 存在内部最优 → BSP 连续混合调度成立")
        print(f"     建议继续补 α=0.25 与 0.75 画完整曲线")
    elif rho > 0:
        print("  ⚠️  单调递增（ρ∈(0,1]）→ α=1 已近最优")
        print("     BSP 退化为 TPG 二值门控；可补 α=0.75 确认单调性")
    else:
        print("  ❌ 混合反而更差 → 支持『伪标签应全量或不用』的 Bang-Bang 策略")
        print("     BSP 仍可写成『相变型调度』，但方法价值主要在 TPG 门控")
    print("-" * 66)


if __name__ == "__main__":
    main()
