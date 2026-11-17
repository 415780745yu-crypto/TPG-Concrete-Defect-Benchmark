# -*- coding: utf-8 -*-
"""46: BSP 试点执行器 —— Stage 0（shot40 补完）+ Stage 1（img100 α=0.5）。

串行执行，避免 6GB 显存并发 OOM。

Stage 0  补完 shot40·seed42
   · A-PL 已训练但未评估 → 补评估
   · repeat-A 未构建未训练 → 构建+训练+评估
   · 目的：得到一个未参与阈值拟合的独立验证点（教师精度 0.2062）

Stage 1  img100·seed42 α=0.5（BSP 定生死）
   · 已构建 data/pseudo_pl_mix/img100_seed42_a0.5
   · 训练+评估，与 seed42 自身配对基线比较：
       α=0   repeat-A = 0.3749
       α≈1  A-PL     = 0.4130
     若 0.5 > 0.4130 → 存在内部最优，BSP 成立

用法：
  python scripts/46_run_bsp_pilot.py                 # 两个阶段都跑
  python scripts/46_run_bsp_pilot.py --stage 1       # 只跑 Stage 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SRC_W = "runs/pw_yolo_source_pretrain/weights/best.pt"


def run(script, args, tag):
    print(f"  [{tag}] {Path(script).name}", flush=True)
    r = subprocess.run([PY, str(ROOT / "scripts" / script)] + args,
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    ok = r.returncode == 0
    print(f"    {'✓' if ok else '✗ 退出码 ' + str(r.returncode)}", flush=True)
    if not ok:
        print("    " + ((r.stderr or r.stdout or "")[-500:]), flush=True)
    return ok


def best_pt(tag):
    return ROOT / "runs" / f"pw_yolo_{tag}" / "weights" / "best.pt"


def stage0():
    print("\n" + "=" * 60)
    print("Stage 0：补完 shot40·seed42（TPG 独立验证点）")
    print("=" * 60, flush=True)

    # A-PL 已训练，补评估
    if best_pt("shot40_seed42_pl").exists():
        run("04_eval.py", [
            "--weights", str(best_pt("shot40_seed42_pl")),
            "--data", "data/damcrack.yaml", "--tag", "shot40_seed42_pl",
            "--protocol", "crossdomain", "--note", "shot40 真标注+伪标签 τ=0.25",
        ], "A-PL 评估")

    # repeat-A：构建 + 训练 + 评估
    run("36_build_repeatA.py", ["--seeds", "seed42", "--tiers", "shot40"], "构建 repeat-A")
    if not best_pt("shot40_seed42_rpt").exists():
        run("03_train.py", [
            "--data", "data/pseudo_pl_rpt/shot40_seed42/data.yaml",
            "--weights", SRC_W, "--epochs", "50", "--tag", "shot40_seed42_rpt",
        ], "repeat-A 训练")
    if best_pt("shot40_seed42_rpt").exists():
        run("04_eval.py", [
            "--weights", str(best_pt("shot40_seed42_rpt")),
            "--data", "data/damcrack.yaml", "--tag", "shot40_seed42_rpt",
            "--protocol", "crossdomain", "--note", "shot40 真标注重复至等量",
        ], "repeat-A 评估")


def stage1(alpha=0.5):
    print("\n" + "=" * 60)
    print(f"Stage 1：img100·seed42 α={alpha}（BSP 判定）")
    print("=" * 60, flush=True)

    ddir = ROOT / "data" / "pseudo_pl_mix" / f"img100_seed42_a{alpha:g}"
    if not ddir.exists():
        run("45_build_mix_dataset.py",
            ["--tier", "img100", "--seed", "seed42", "--alpha", str(alpha)], "构建混合集")

    tag = f"img100_seed42_mix{alpha:g}"
    if not best_pt(tag).exists():
        run("03_train.py", [
            "--data", f"data/pseudo_pl_mix/img100_seed42_a{alpha:g}/data.yaml",
            "--weights", SRC_W, "--epochs", "50", "--tag", tag,
        ], "混合集训练")
    if best_pt(tag).exists():
        run("04_eval.py", [
            "--weights", str(best_pt(tag)), "--data", "data/damcrack.yaml",
            "--tag", tag, "--protocol", "crossdomain",
            "--note", f"img100 α={alpha:g} 混合比例",
        ], "混合集评估")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "1", "all"], default="all")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    if args.stage in ("0", "all"):
        stage0()
    if args.stage in ("1", "all"):
        stage1(args.alpha)

    print("\n" + "=" * 60)
    print("完成。结果：")
    print("=" * 60)
    import csv
    p = ROOT / "results" / "crossdomain.csv"
    with open(p, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "shot40" in r["tag"] or "mix" in r["tag"]:
                print(f"  {r['tag']:<28} {r['map50']}")


if __name__ == "__main__":
    main()
