# -*- coding: utf-8 -*-
"""run_plan_A.py —— 方案 A（伪标签自训练）一键复现脚本。

完整流程（7 档 seed42）：
  1. 28_pseudo_label.py      各档 A 组 best.pt 打伪标签（τ=0.25）
  2. 29_merge_pseudo.py      合并真标注 + 伪标签 -> data/pseudo_pl/<tier>_seed42/
  3. 03_train.py             逐档训练（起点 source_pretrain，50 epochs，seed 42）
  4. 33_auto_eval_watch.py   自动评估完成的档位 -> crossdomain.csv
  5. 32_pl_report_v2.py      Δ 曲线（x 轴 = 真标注框数）+ 分档柱状图
  6. 30_pl_report.py         兼容旧版 Δ 表/图

用法：
  python scripts/run_plan_A.py --stage all          # 全流程（含等待训练）
  python scripts/run_plan_A.py --stage pseudo       # 仅 1+2
  python scripts/run_plan_A.py --stage train        # 仅 3
  python scripts/run_plan_A.py --stage eval         # 仅 4+5+6

前置：A 组 7 档 best.pt 已存在（runs/pw_yolo_<tier>_seed42/weights/best.pt）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
PY = sys.executable
S = ROOT / "scripts"


def run(script: str, *extra: str) -> int:
    cmd = [PY, str(S / script), *extra]
    print(f"\n=== {script} {' '.join(extra)} ===")
    return subprocess.run(cmd).returncode


def stage_pseudo() -> None:
    run("28_pseudo_label.py")
    run("29_merge_pseudo.py")


def stage_train() -> None:
    src = "runs/pw_yolo_source_pretrain/weights/best.pt"
    for t in TIERS:
        best = ROOT / "runs" / f"pw_yolo_{t}_seed42_pl" / "weights" / "best.pt"
        last = ROOT / "runs" / f"pw_yolo_{t}_seed42_pl" / "weights" / "last.pt"
        if best.exists() and last.exists():
            # 已有产物：保守跳过（人工确认是否重训）
            print(f"[SKIP] {t}: 已存在 best.pt/last.pt")
            continue
        run("03_train.py",
            "--data", f"data/pseudo_pl/{t}_seed42/data.yaml",
            "--weights", src, "--epochs", "50", "--tag", f"{t}_seed42_pl")


def stage_eval() -> None:
    # 单轮评估已完成的档位（不阻塞等待）
    run("33_auto_eval_watch.py", "--stale-min", "5")
    run("32_pl_report_v2.py")
    run("30_pl_report.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "pseudo", "train", "eval"])
    args = ap.parse_args()

    if args.stage in ("all", "pseudo"):
        stage_pseudo()
    if args.stage in ("all", "train"):
        stage_train()
    if args.stage in ("all", "eval"):
        stage_eval()
    print("\n[run_plan_A] done")


if __name__ == "__main__":
    main()
