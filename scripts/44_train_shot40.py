# -*- coding: utf-8 -*-
"""44: 训练 shot40 验证档（TPG 门控的独立验证点）。

对 shot40 的 3 个标注抽样种子，各训练三组：
  A      = 仅真标注
  A-PL   = 真标注 + 伪标签（τ=0.25）
  repeat-A = 真标注重复至与 A-PL 等量

用途：验证「教师精度 → 净增益」的分段门控逻辑。
      shot30（教师 0.1920，净 +0.0154）与 shot50（教师 0.2600，净 +0.0372）
      之间应有一个中间点，用于检验 TPG 在 τ=0.25 分档内的单调性。

用法：
  python scripts/44_train_shot40.py            # 全部
  python scripts/44_train_shot40.py --seeds 42 # 单种子
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = (42, 43, 44)


def run(script, args, tag):
    print(f"  [{tag}] {Path(script).name} {' '.join(args)}", flush=True)
    r = subprocess.run([PY, str(ROOT / "scripts" / script)] + args,
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(f"    ⚠️ 退出码 {r.returncode}")
        tail = (r.stderr or r.stdout or "")[-400:]
        print(f"    {tail}")
    return r.returncode


def best_pt(tag):
    return ROOT / "runs" / f"pw_yolo_{tag}" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", default="50")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tier = "shot40"

    for s in seeds:
        seed = f"seed{s}"
        print(f"\n{'='*66}")
        print(f"shot40 · {seed}")
        print("=" * 66)

        # ---------- 1. A 组：仅真标注 ----------
        tag_a = f"{tier}_{seed}"
        if not best_pt(tag_a).exists():
            run("03_train.py", [
                "--data", f"data/fewshot/{tier}_{seed}.yaml",
                "--weights", "runs/pw_yolo_source_pretrain/weights/best.pt",
                "--epochs", args.epochs, "--tag", tag_a,
            ], "A 训练")
        else:
            print(f"  [A] 已存在，跳过训练")
        if best_pt(tag_a).exists():
            run("04_eval.py", [
                "--weights", str(best_pt(tag_a)), "--data", "data/damcrack.yaml",
                "--tag", tag_a, "--protocol", "crossdomain", "--note", "shot40 仅真标注",
            ], "A 评估")

        # ---------- 2. A-PL：伪标签（用该档自举权重生成） ----------
        tag_pl = f"{tier}_{seed}_pl"
        pdir = ROOT / "results" / "predictions" / f"pseudo_{tier}_{seed}"
        if not (pdir / "labels").exists():
            run("28_pseudo_label.py", ["--seed", seed, "--tiers", tier], "生成伪标签")
        run("29_merge_pseudo.py", ["--seed", seed, "--tiers", tier], "合并伪标签")
        if not best_pt(tag_pl).exists():
            run("03_train.py", [
                "--data", f"data/pseudo_pl/{tier}_{seed}/data.yaml",
                "--weights", "runs/pw_yolo_source_pretrain/weights/best.pt",
                "--epochs", args.epochs, "--tag", tag_pl,
            ], "A-PL 训练")
        else:
            print(f"  [A-PL] 已存在，跳过训练")
        if best_pt(tag_pl).exists():
            run("04_eval.py", [
                "--weights", str(best_pt(tag_pl)), "--data", "data/damcrack.yaml",
                "--tag", tag_pl, "--protocol", "crossdomain",
                "--note", "shot40 真标注+伪标签 τ=0.25",
            ], "A-PL 评估")

        # ---------- 3. repeat-A：真标注重复至等量 ----------
        tag_rp = f"{tier}_{seed}_rpt"
        run("36_build_repeatA.py", ["--seed", seed, "--tiers", tier], "构建 repeat-A")
        if not best_pt(tag_rp).exists():
            run("03_train.py", [
                "--data", f"data/pseudo_pl_rpt/{tier}_{seed}/data.yaml",
                "--weights", "runs/pw_yolo_source_pretrain/weights/best.pt",
                "--epochs", args.epochs, "--tag", tag_rp,
            ], "repeat-A 训练")
        else:
            print(f"  [repeat-A] 已存在，跳过训练")
        if best_pt(tag_rp).exists():
            run("04_eval.py", [
                "--weights", str(best_pt(tag_rp)), "--data", "data/damcrack.yaml",
                "--tag", tag_rp, "--protocol", "crossdomain",
                "--note", "shot40 真标注重复至等量（训练量对照）",
            ], "repeat-A 评估")

    print(f"\n{'='*66}")
    print("完成。请运行以下命令查看结果：")
    print("  python -c \"import csv; [print(r['tag'], r['map50']) for r in csv.DictReader(open('results/crossdomain.csv',encoding='utf-8-sig')) if 'shot40' in r['tag']]\"")


if __name__ == "__main__":
    main()
