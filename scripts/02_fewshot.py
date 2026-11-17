"""阶段 3：few-shot 采样（支持两种单位：instances_per_class / images）。

instances_per_class（默认，行为与旧版完全一致）：
  每类 N 个带框实例 × 3 seeds，输出 data/fewshot/shot{N}_seed{S}.txt/.yaml，
  索引 results/fewshot_index.csv。

images（新增）：
  图像级采样：打乱 damcrack_train 顺序，取「至少含 1 个目标框」的图直到取满 N 张；
  若任一类出现的图像数 < 5，换 seed+1000 重抽，索引记录实际 seed；
  输出 data/<out-subdir>/img{N}_seed{S}.txt/.yaml，索引 results/fewshot_image_index.csv。
  val/test 仍用完整划分。

用法：
  python scripts/02_fewshot.py                                   # 默认：instances_per_class，读配置
  python scripts/02_fewshot.py --unit images --shots "100,200,500" --seeds "42"
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from common import ROOT, append_csv, load_cfg, need, read_yolo_label, write_ultralytics_yaml


def sample_instances(n_shot, seed, per_image, names):
    rng = random.Random(seed)
    order = list(per_image.keys())
    rng.shuffle(order)
    chosen, tally = [], defaultdict(int)
    for im in order:
        if all(tally.get(c, 0) >= n_shot for c in names):
            break
        for c, cnt in per_image[im].items():
            if c in names and tally[c] < n_shot:
                tally[c] += cnt
        chosen.append(im)
    return chosen, {c: tally[c] for c in names}


def sample_images(n_img, seed, per_image, names):
    """图像级采样：返回 (chosen, class_img_count, instances)。"""
    rng = random.Random(seed)
    order = list(per_image.keys())
    rng.shuffle(order)
    chosen = []
    class_img = defaultdict(int)
    instances = defaultdict(int)
    for im in order:
        cnt = per_image[im]
        if not any(cnt.get(c, 0) > 0 for c in names):
            continue
        chosen.append(im)
        for c in names:
            n = cnt.get(c, 0)
            if n > 0:
                class_img[c] += 1
            instances[c] += n
        if len(chosen) >= n_img:
            break
    return chosen, dict(class_img), {c: instances[c] for c in names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["instances_per_class", "images"], default="instances_per_class")
    ap.add_argument("--shots", default="", help="逗号分隔，如 100,200,500；空则读配置")
    ap.add_argument("--seeds", default="", help="逗号分隔，如 42；空则读配置")
    ap.add_argument("--out-subdir", default=None, help="images 模式输出子目录名（默认 fewshot_img）")
    args = ap.parse_args()

    cfg = load_cfg()
    fs = cfg["fewshot"]
    unit = args.unit
    shots = [int(x) for x in args.shots.split(",") if x.strip()] if args.shots.strip() else fs["shots"]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds.strip() else fs["seeds"]

    train_txt = ROOT / "data" / "splits" / "damcrack_train.txt"
    val_txt = ROOT / "data" / "splits" / "damcrack_val.txt"
    test_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    need(train_txt.exists(), "请先运行 01_prepare_data.py 生成划分文件")

    dc = cfg["datasets"]["damcrack"]
    root = Path(dc["root"]) if Path(dc["root"]).is_absolute() else ROOT / dc["root"]
    lbl_dir = root / dc["label_dir"]
    images = [Path(p) for p in train_txt.read_text(encoding="utf-8").splitlines() if p.strip()]

    per_image = {}
    for im in images:
        lb = lbl_dir / f"{im.stem}.txt"
        need(lb.exists(), f"缺少标签 {lb}")
        per_image[im] = dict(read_yolo_label(lb))

    names = cfg["classes"]["target"]

    if unit == "instances_per_class":
        out_dir = ROOT / "data" / "fewshot"
        out_dir.mkdir(parents=True, exist_ok=True)
        for n_shot in shots:
            for seed in seeds:
                chosen, actual = sample_instances(n_shot, seed, per_image, names)
                need(all(v >= n_shot for v in actual.values()),
                     f"{n_shot}-shot/seed={seed}: 训练集实例不足，实际 {actual}，请降低 shot 数或检查类别映射")
                subset = out_dir / f"shot{n_shot}_seed{seed}.txt"
                subset.write_text("\n".join(p.as_posix() for p in chosen) + "\n", encoding="utf-8")
                write_ultralytics_yaml(out_dir / f"shot{n_shot}_seed{seed}.yaml", root,
                                       {"train": subset, "val": val_txt, "test": test_txt}, names)
                append_csv("fewshot_index.csv", {
                    "shots": n_shot, "seed": seed, "unit": unit,
                    "images": len(chosen), "actual_instances_crack": actual[0],
                    "actual_instances_spalling": actual[1], "yaml": subset.with_suffix(".yaml").name,
                })
                print(f"[ok] {n_shot}-shot seed={seed}: {len(chosen)} 张图, 实例 {dict(actual)}")
    else:
        out_dir = ROOT / "data" / (args.out_subdir or "fewshot_img")
        out_dir.mkdir(parents=True, exist_ok=True)
        for n_img in shots:
            for seed in seeds:
                actual_seed = seed
                chosen = class_img = instances = None
                while True:
                    chosen, class_img, instances = sample_images(n_img, actual_seed, per_image, names)
                    if all(class_img.get(c, 0) >= 5 for c in names):
                        break
                    actual_seed += 1000
                    need(actual_seed - seed <= 10000, f"{n_img}-img seed={seed}: 重抽超限仍未满足每类>=5 张")
                subset = out_dir / f"img{n_img}_seed{seed}.txt"
                subset.write_text("\n".join(p.as_posix() for p in chosen) + "\n", encoding="utf-8")
                write_ultralytics_yaml(out_dir / f"img{n_img}_seed{seed}.yaml", root,
                                       {"train": subset, "val": val_txt, "test": test_txt}, names)
                append_csv("fewshot_image_index.csv", {
                    "shots": n_img, "seed": actual_seed, "unit": unit,
                    "images": len(chosen), "actual_instances_crack": instances[0],
                    "actual_instances_spalling": instances[1], "yaml": subset.with_suffix(".yaml").name,
                })
                print(f"[ok] {n_img}-img seed={seed}(实际{actual_seed}): {len(chosen)} 张图, "
                      f"每类图像 {class_img}, 实例 {instances}")

    print("[done] 结果由 03/04 脚本写入 results/crossdomain.csv")


if __name__ == "__main__":
    main()
