# -*- coding: utf-8 -*-
"""31: 对 7 档 A-PL best.pt 统一评估（--protocol crossdomain）。

在 30_pl_report.py 之前运行。评估结果写入 crossdomain.csv（tag=<tier>_seed42_pl）。

运行：
  python scripts/31_eval_pl.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=TIERS)
    args = ap.parse_args()

    py = sys.executable
    for tier in args.tiers:
        best = ROOT / "runs" / f"pw_yolo_{tier}_seed42_pl" / "weights" / "best.pt"
        if not best.exists():
            print(f"[skip] {tier}: 缺 {best}")
            continue
        tag = f"{tier}_seed42_pl"
        print(f"[eval] {tag}")
        r = subprocess.run(
            [py, str(ROOT / "scripts" / "04_eval.py"),
             "--weights", str(best),
             "--data", str(ROOT / "data" / "damcrack.yaml"),
             "--tag", tag, "--protocol", "crossdomain",
             "--note", "pseudo-label"],
        )
        if r.returncode != 0:
            print(f"[FAIL] {tag} exit={r.returncode}")
    print("[ALL EVAL DONE] 下一步 python scripts/30_pl_report.py")


if __name__ == "__main__":
    main()
