# -*- coding: utf-8 -*-
"""45: 构建「真标注 : 伪标签」混合比例训练集（BSP 的 α 调度实验）。

背景：
  现有两端点已测——
    α=0   : repeat-A（真标注重复至与 A-PL 等行数，全真标签）
    α≈1   : A-PL（真标注 + 伪标签，伪标签占约 99%）
  BSP 要回答的是：中间比例是否更优？为此需构造 α∈(0,1) 的混合训练集。

实现要点：
  · 总行数 T 与 A-PL 完全一致 → 步数匹配，唯一变量是混合比例
  · α·T 行来自伪标签池，(1−α)·T 行来自真标注图（必要时循环重复）
  · 伪标签池与真标注子集天然不相交（池定义已扣除标注子集），标签不冲突
  · 图片目录用 junction 指向原图目录，不改动原始数据

产物：data/pseudo_pl_mix/<tier>_<seed>_a<alpha>/{images,labels,train.txt,data.yaml}

用法：
  python scripts/45_build_mix_dataset.py --tier img100 --seed seed42 --alpha 0.5
  python scripts/45_build_mix_dataset.py --tier img100 --seed seed42 --alpha 0.25 0.5 0.75
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import ROOT, need

IMG_SRC = "D:/YaLi/Damage Detection/images"
GT_LABEL_SRC = "D:/YaLi/Damage Detection/labels/Yolo"
VAL_TXT = ROOT / "data" / "splits" / "damcrack_val.txt"
TEST_TXT = ROOT / "data" / "splits" / "damcrack_test.txt"
NAMES = {"0": "crack", "1": "spalling"}


def make_junction(link: Path, target: str) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), target],
                       capture_output=True, text=True)
    need(link.exists(), f"junction 创建失败: {r.stdout} {r.stderr}")


def gt_subset(tier: str, seed: str) -> list[str]:
    for base in ("data/fewshot", "data/fewshot_img"):
        p = ROOT / base / f"{tier}_{seed}.txt"
        if p.exists():
            return [Path(l.strip()).name for l in
                    p.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise FileNotFoundError(f"找不到 {tier}_{seed}.txt")


def pseudo_label_dir(tier: str, seed: str) -> Path:
    base = ROOT / "results" / "predictions" / f"pseudo_{tier}_{seed}"
    for cand in (base / "labels" / "labels", base / "labels"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"找不到伪标签目录: {base}")


def build(tier: str, seed: str, alpha: float) -> Path:
    gt_names = gt_subset(tier, seed)
    gt_stems = {Path(n).stem for n in gt_names}

    apl_train = ROOT / "data" / "pseudo_pl" / f"{tier}_{seed}" / "train.txt"
    need(apl_train.exists(), f"缺 {apl_train}，先跑 29_merge_pseudo.py")
    apl_lines = [l.strip() for l in
                 apl_train.read_text(encoding="utf-8").splitlines() if l.strip()]
    T = len(apl_lines)

    # 伪标签池（排除真标注图，双保险）
    pdir = pseudo_label_dir(tier, seed)
    pseudo_stems = sorted(p.stem for p in pdir.glob("*.txt")
                          if p.stat().st_size > 0 and p.stem not in gt_stems)
    need(pseudo_stems, f"{tier}_{seed} 伪标签池为空")

    n_pseudo_rows = int(round(alpha * T))
    n_real_rows = T - n_pseudo_rows
    need(n_pseudo_rows <= len(pseudo_stems),
         f"伪标签池不足: 需要 {n_pseudo_rows}，仅 {len(pseudo_stems)}")

    # 抽样伪标签（固定顺序，可复现）
    pseudo_pick = pseudo_stems[:n_pseudo_rows]
    # 真标注循环重复
    real_rows = [gt_names[i % len(gt_names)] for i in range(n_real_rows)]

    out = ROOT / "data" / "pseudo_pl_mix" / f"{tier}_{seed}_a{alpha:g}"
    if out.exists():
        shutil.rmtree(out)
    idir, ldir = out / "images", out / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    make_junction(idir, IMG_SRC)

    # 真标注标签
    for n in set(real_rows):
        src = Path(GT_LABEL_SRC) / f"{Path(n).stem}.txt"
        need(src.exists(), f"缺真标注 {src}")
        shutil.copy2(src, ldir / f"{Path(n).stem}.txt")
    # 伪标签
    for st in pseudo_pick:
        shutil.copy2(pdir / f"{st}.txt", ldir / f"{st}.txt")

    # 训练清单：先真后伪再打散（顺序对 Ultralytics 无影响，但保持可读）
    lines = [str(idir / n) for n in real_rows] + \
            [str(idir / f"{st}.jpg") for st in pseudo_pick]
    need(len(lines) == T, f"[BUG] 行数 {len(lines)} != {T}")
    (out / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "data.yaml").write_text("\n".join([
        f"path: {out.as_posix()}", "train: train.txt",
        f"val: {VAL_TXT.as_posix()}", f"test: {TEST_TXT.as_posix()}",
        f"nc: {len(NAMES)}", "names:"] +
        [f"  {k}: {v}" for k, v in sorted(NAMES.items())]) + "\n", encoding="utf-8")

    print(f"[build] {tier}_{seed} α={alpha:g}: "
          f"{n_real_rows} 真标注行（{len(gt_names)} 张图循环）+ "
          f"{n_pseudo_rows} 伪标签行 = {T} 行")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--alpha", nargs="+", type=float, required=True)
    args = ap.parse_args()

    need(Path(IMG_SRC).is_dir(), f"原图目录不存在: {IMG_SRC}")
    for a in args.alpha:
        need(0.0 <= a <= 1.0, f"α 必须在 [0,1]: {a}")
        build(args.tier, args.seed, a)


if __name__ == "__main__":
    main()
