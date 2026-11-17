# -*- coding: utf-8 -*-
"""run_plan_A_seeds.py —— 批量跑方案 A 的 seed43/44 全流程。

对每个 seed 依序执行：28 打伪标签 -> 29 合并 -> 7 档训练 -> 逐档评估。
每完成一档评估即去重写入 crossdomain.csv，便于中途检查。

用法：
  python scripts/run_plan_A_seeds.py --seeds seed43 seed44
  python scripts/run_plan_A_seeds.py --seeds seed43 --tiers shot10   # 单档调试
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from common import ROOT

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
PY = sys.executable
S = ROOT / "scripts"


def run(script: str, *extra: str) -> int:
    cmd = [PY, str(S / script), *extra]
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    return subprocess.run(cmd).returncode


def _already_evaluated(tag: str) -> bool:
    """crossdomain.csv 中是否已有该 tag 且 map50 非空（幂等判据）。"""
    import csv as _csv
    p = ROOT / "results" / "crossdomain.csv"
    if not p.exists():
        return False
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            if r.get("tag", "").strip() == tag and (r.get("map50") or "").strip():
                return True
    return False


def _dedup() -> None:
    """去除 crossdomain.csv 重复 tag（保留最后一条），保持无引号格式。"""
    import csv as _csv
    p = ROOT / "results" / "crossdomain.csv"
    if not p.exists():
        return
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return
    fields = list(rows[0].keys())
    seen = {}
    for r in rows:
        seen[r["tag"].strip()] = r
    if len(seen) == len(rows):
        return
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=fields, quoting=_csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(seen.values())
    print(f"[dedup] crossdomain.csv {len(rows)} -> {len(seen)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["seed43", "seed44"])
    ap.add_argument("--tiers", nargs="*", default=TIERS)
    args = ap.parse_args()

    for seed in args.seeds:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}", flush=True)

        # 1) 打伪标签
        rc = run("28_pseudo_label.py", "--seed", seed, "--tiers", *args.tiers)
        if rc != 0:
            print(f"[FAIL] {seed} 28_pseudo_label exit={rc}", flush=True)
            continue

        # 2) 合并
        rc = run("29_merge_pseudo.py", "--seed", seed, "--tiers", *args.tiers)
        if rc != 0:
            print(f"[FAIL] {seed} 29_merge_pseudo exit={rc}", flush=True)
            continue

        # 3) 训练 + 4) 评估（逐档，评估后立即去重）
        for t in args.tiers:
            tag = f"{t}_{seed}_pl"
            best = ROOT / "runs" / f"pw_yolo_{tag}" / "weights" / "best.pt"
            if not best.exists():
                rc = run("03_train.py",
                         "--data", f"data/pseudo_pl/{t}_{seed}/data.yaml",
                         "--weights", "runs/pw_yolo_source_pretrain/weights/best.pt",
                         "--epochs", "50", "--tag", tag)
                if rc != 0:
                    print(f"[FAIL] {seed} train {t} exit={rc}", flush=True)
                    continue
            else:
                print(f"[SKIP] {t} 训练已完成", flush=True)
            # 幂等：已在 crossdomain.csv 且有值则跳过评估
            if not _already_evaluated(tag):
                run("04_eval.py", "--weights", str(best),
                    "--data", "data/damcrack.yaml", "--tag", tag,
                    "--protocol", "crossdomain", "--note", "pseudo-label")
                _dedup()
            else:
                print(f"[SKIP] {tag} 已评估，跳过", flush=True)

    print("\n[ALL SEEDS DONE]", flush=True)


if __name__ == "__main__":
    main()
