# -*- coding: utf-8 -*-
"""36: 构建「训练量匹配」对照集（repeat-A）。

动机（应对审稿质疑）：
  A-PL 组的训练集远大于 A 组（shot10: 5 图 → 547 图），因此总梯度更新步数是
  A 组的 5–103 倍。若不做匹配，"Δ 来自伪标签监督"与"Δ 来自训练更久/见更多样本"
  两个解释无法分离。

本脚本把 A 组的真标注子集**重复列进 train.txt**，使其图数与 A-PL 组一致：
  - 图数、epoch、每 epoch 批次数、总步数 均与 A-PL 匹配
  - 唯一差异：额外样本携带的是**真标签**（repeat-A）还是**伪标签**（A-PL）
  - 已实测确认 Ultralytics 的 get_img_files() 对 train.txt 行**不做去重**，
    故重复列同一路径即可有效重复采样

产物：data/pseudo_pl_rpt/<tier>_seed<s>/{images(junction), labels, train.txt, data.yaml}

用法：
  python scripts/36_build_repeatA.py --seeds seed42
  python scripts/36_build_repeatA.py --seeds seed42 seed43 seed44
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import ROOT, need

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]
IMG_SRC = "D:/YaLi/Damage Detection/images"
GT_LABEL_SRC = "D:/YaLi/Damage Detection/labels/Yolo"
VAL_TXT = ROOT / "data" / "splits" / "damcrack_val.txt"
TEST_TXT = ROOT / "data" / "splits" / "damcrack_test.txt"


def labeled_subset(tier: str, seed: str) -> Path:
    for base in ("data/fewshot", "data/fewshot_img"):
        p = ROOT / base / f"{tier}_{seed}.txt"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {tier}_{seed}.txt")


def make_junction(link: Path, target: str) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), target],
                       capture_output=True, text=True)
    need(link.exists(), f"junction 创建失败: {r.stdout} {r.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["seed42"])
    ap.add_argument("--tiers", nargs="*", default=TIERS)
    args = ap.parse_args()

    need(Path(IMG_SRC).is_dir(), f"原图目录不存在: {IMG_SRC}")
    need(Path(GT_LABEL_SRC).is_dir(), f"真标注目录不存在: {GT_LABEL_SRC}")

    for seed in args.seeds:
        for tier in args.tiers:
            sub = labeled_subset(tier, seed)
            gt_names = [Path(p.strip()).name for p in
                        sub.read_text(encoding="utf-8").splitlines() if p.strip()]
            target_train = ROOT / "data" / "pseudo_pl" / f"{tier}_{seed}" / "train.txt"
            need(target_train.exists(), f"缺 {target_train}，先跑 29_merge_pseudo.py")
            n_target = len([l for l in target_train.read_text(encoding="utf-8").splitlines() if l.strip()])

            # 计算重复次数：重复列出真标注图，使总行数 >= A-PL 图数
            k = max(1, -(-n_target // len(gt_names)))  # ceil
            lines = []
            for _ in range(k):
                lines.extend(gt_names)
            lines = lines[:n_target] if len(lines) > n_target else lines

            out = ROOT / "data" / "pseudo_pl_rpt" / f"{tier}_{seed}"
            if out.exists():
                shutil.rmtree(out)
            idir, ldir = out / "images", out / "labels"
            ldir.mkdir(parents=True, exist_ok=True)
            make_junction(idir, IMG_SRC)

            for n in gt_names:
                src = Path(GT_LABEL_SRC) / (Path(n).stem + ".txt")
                need(src.exists(), f"缺真标注 {src}")
                shutil.copy2(src, ldir / (Path(n).stem + ".txt"))

            # ⚠️ 必须用 Windows 原生反斜杠（Ultralytics 靠 os.sep 做 images->labels 替换）
            (out / "train.txt").write_text(
                "\n".join(str(idir / n) for n in lines) + "\n", encoding="utf-8")

            names = {"0": "crack", "1": "spalling"}
            (out / "data.yaml").write_text("\n".join([
                f"path: {out.as_posix()}", "train: train.txt",
                f"val: {VAL_TXT.as_posix()}", f"test: {TEST_TXT.as_posix()}",
                f"nc: {len(names)}", "names:"] +
                [f"  {kk}: {vv}" for kk, vv in sorted(names.items())]) + "\n", encoding="utf-8")

            actual = len((out / "train.txt").read_text(encoding="utf-8").splitlines())
            need(actual == n_target, f"[BUG] {tier} train.txt {actual} != {n_target}")
            print(f"[build] {tier}_{seed}: {len(gt_names)} 张真标注 x {k} 次 = {actual} 行 "
                  f"(与 A-PL 的 {n_target} 行匹配)")


if __name__ == "__main__":
    main()
