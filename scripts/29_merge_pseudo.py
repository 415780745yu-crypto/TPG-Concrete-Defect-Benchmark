# -*- coding: utf-8 -*-
"""29: 合并真标注 + 伪标签，生成方案 A 的合并数据集（零图片拷贝）。

方案 A（伪标签自训练）第 2 步。τ 锁死 0.25（见 28_pseudo_label.py）。

布局（Ultralytics 靠字符串替换 images/ -> labels/ 推标签路径）：
  data/pseudo_pl/<tier>_seed42/
      images/   -> 目录 junction，指向 D:/YaLi/Damage Detection/images（零拷贝）
      labels/   -> 合并后的标签 txt（真标注 + 伪标签，仅几十 KB）
      train.txt -> 仅列出「有框」的图，格式 .../images/Axxxx.jpg
      data.yaml -> path 指向本目录，train: train.txt，val/test 沿用原划分

train.txt 行数 = 该档真标注图数 + 有效伪标签图数，可用文本编辑器直接核对（零成本自检）。

运行：
  python scripts/29_merge_pseudo.py                 # 全 7 档
  python scripts/29_merge_pseudo.py --tiers shot50 # 单档调试
  python scripts/29_merge_pseudo.py --dry-run      # 只打印统计，不写文件
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from common import ROOT, RESULTS, load_cfg, need

IMG_SRC = "D:/YaLi/Damage Detection/images"      # 原图目录（实测）
GT_LABEL_SRC = "D:/YaLi/Damage Detection/labels/Yolo"  # 真标注（实测）
VAL_TXT = ROOT / "data" / "splits" / "damcrack_val.txt"
TEST_TXT = ROOT / "data" / "splits" / "damcrack_test.txt"

TIERS = [
    "shot10", "shot20", "shot30", "shot50",
    "img100", "img200", "img500",
]


def labeled_subset(tier: str, seed: str) -> Path:
    for base in ("data/fewshot", "data/fewshot_img"):
        p = ROOT / base / f"{tier}_{seed}.txt"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {tier}_{seed}.txt")


def pseudo_dir(tier: str, seed: str, suffix: str = "") -> Path:
    """28 脚本的预测输出目录（predict_to_txt 返回 labels/labels 或 labels）。"""
    d = RESULTS / "predictions" / f"pseudo_{tier}_{seed}{suffix}" / "labels"
    cand = d / "labels"
    return cand if cand.exists() else d


def make_junction(link: Path, target: str) -> None:
    """跨盘目录 junction（mklink /J），不需管理员权限。已实测可用。"""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), target],
                       capture_output=True, text=True)
    need(link.exists(), f"junction 创建失败: {r.stdout} {r.stderr}")


def read_pseudo_lines(txt: Path) -> list[str]:
    """读伪标签 txt，只保留 cid∈(0,1)，输出标准 5 列 cid cx cy w h。"""
    out = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        try:
            cid = int(float(p[0]))
        except ValueError:
            continue
        if cid not in (0, 1):        # 丢弃 moisture 幻影类
            continue
        out.append(f"{cid} {p[1]} {p[2]} {p[3]} {p[4]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=TIERS)
    ap.add_argument("--seed", default="seed42", help="抽样种子，默认 seed42")
    ap.add_argument("--suffix", default="", help="伪标签目录后缀（须与 28 一致）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()
    seed = args.seed

    cfg = load_cfg()
    names = cfg["classes"]["target"]
    need(Path(IMG_SRC).is_dir(), f"原图目录不存在: {IMG_SRC}")
    need(Path(GT_LABEL_SRC).is_dir(), f"真标注目录不存在: {GT_LABEL_SRC}")

    rows = []
    for tier in args.tiers:
        sub = labeled_subset(tier, seed)
        gt_imgs = [Path(p.strip()).name for p in sub.read_text(encoding="utf-8").splitlines() if p.strip()]
        gt_set = set(gt_imgs)

        pdir = pseudo_dir(tier, seed, args.suffix)
        need(pdir.exists(), f"缺伪标签目录 {pdir}，先跑 28_pseudo_label.py")

        # 收集有效伪标签（有框、且不在真标注子集中——防泄漏/防重复）
        pseudo = {}
        for txt in pdir.glob("*.txt"):
            if txt.stem + ".jpg" in gt_set:
                continue                      # 该图属于真标注，跳过
            lines = read_pseudo_lines(txt)
            if lines:                         # 无框图丢弃（不引入负样本）
                pseudo[txt.stem + ".jpg"] = lines

        overlap = gt_set & set(pseudo)
        need(not overlap, f"[BUG] {tier} 真标注与伪标签重叠 {len(overlap)} 张")

        total = len(gt_imgs) + len(pseudo)
        print(f"[stat] {tier}: 真标注 {len(gt_imgs)} + 有效伪标签 {len(pseudo)} = train.txt {total} 行")

        if args.dry_run:
            rows.append({"tier": tier, "n_gt": len(gt_imgs), "n_pseudo": len(pseudo),
                         "n_train": total, "tag": f"{tier}_{seed}_pl", "map50": "", "note": "pseudo-label"})
            continue

        out = ROOT / "data" / "pseudo_pl" / f"{tier}_{seed}{args.suffix}"
        if out.exists():
            shutil.rmtree(out)
        idir = out / "images"
        ldir = out / "labels"
        ldir.mkdir(parents=True, exist_ok=True)
        make_junction(idir, IMG_SRC)

        # 写标签：真标注（原样复制）+ 伪标签（5 列）
        for n in gt_imgs:
            src = Path(GT_LABEL_SRC) / (Path(n).stem + ".txt")
            need(src.exists(), f"缺真标注 {src}")
            shutil.copy2(src, ldir / (Path(n).stem + ".txt"))
        for n, lines in pseudo.items():
            (ldir / (Path(n).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

        # train.txt：只列有框的图。
        # ⚠️ 必须用 Windows 原生反斜杠路径（str），不能用 .as_posix()：
        # Ultralytics 的 img2label_paths 用 os.sep 构造 "\\images\\" 做字符串替换，
        # 正斜杠 "/images/" 匹配不上会导致标签全部找不到（已在验证中实测）。
        train_lines = [str(idir / n) for n in gt_imgs] + \
                      [str(idir / n) for n in pseudo]
        (out / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

        # 自检：train.txt 行数必须等于统计值
        actual = len((out / "train.txt").read_text(encoding="utf-8").splitlines())
        need(actual == total, f"[BUG] {tier} train.txt 行数 {actual} != {total}")

        (out / "data.yaml").write_text("\n".join([
            f"path: {out.as_posix()}", "train: train.txt",
            f"val: {VAL_TXT.as_posix()}", f"test: {TEST_TXT.as_posix()}",
            f"nc: {len(names)}", "names:"] +
            [f"  {k}: {v}" for k, v in sorted(names.items())]) + "\n", encoding="utf-8")

        rows.append({"tier": tier, "n_gt": len(gt_imgs), "n_pseudo": len(pseudo),
                     "n_train": total, "tag": f"{tier}_{seed}_pl", "map50": "", "note": "pseudo-label"})
        print(f"[write] {out.relative_to(ROOT)}  (images/ 为 junction，零图片拷贝)")

    # 记录到独立 CSV（不覆盖 selftrain.csv）
    if not args.dry_run:
        dst = RESULTS / "pseudo_pl_build.csv"
        with open(dst, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["tier", "n_gt", "n_pseudo", "n_train", "tag", "map50", "note"])
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {dst.relative_to(ROOT)}")

    print("[ALL DONE] 下一步用 03_train.py 训练，例如：")
    print("  python scripts/03_train.py --data data/pseudo_pl/shot50_seed42/data.yaml "
          "--weights runs/pw_yolo_source_pretrain/weights/best.pt --epochs 50 --tag shot50_seed42_pl")


if __name__ == "__main__":
    main()
