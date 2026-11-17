"""阶段 1：校验数据结构、类别映射、母图级（或近重复）划分，生成 Ultralytics data.yaml。

用法：
  python scripts/01_prepare_data.py --dry-run     # 先看统计，不写文件
  python scripts/01_prepare_data.py               # 正式生成
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("[STOP] 缺少 Pillow。请先执行: pip install -r requirements.txt")

from common import (ROOT, RESULTS, abs_path, append_csv, inspect_voc_names, load_cfg,
                    need, read_yolo_label, voc_to_yolo, write_ultralytics_yaml)


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size))
        px = list(g.getdata())
    v = 0
    for r in range(size):
        for c in range(size):
            v = (v << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return v


def popcount(x: int) -> int:
    return bin(x).count("1")


def group_near_duplicate(items: list[Path], threshold: int) -> list[list[Path]]:
    """简单并查集分组：哈希距离小于阈值的切片视为同源，确保不跨划分。"""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = [dhash(p) for p in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if popcount(hashes[i] ^ hashes[j]) <= threshold:
                union(i, j)
    buckets: dict[int, list[Path]] = defaultdict(list)
    for i, p in enumerate(items):
        buckets[find(i)].append(p)
    return list(buckets.values())


def collect(spec: dict, key: str) -> tuple[list[Path], dict]:
    root = abs_path(spec["root"])
    img_dir = root / spec["image_dir"]
    lbl_dir = root / spec["label_dir"]
    need(img_dir.is_dir(), f"{key}: 图像目录不存在或配置未填 -> {img_dir}")
    need(lbl_dir.is_dir(), f"{key}: 标签目录不存在或配置未填 -> {lbl_dir}")
    fmt = spec.get("label_format")
    need(fmt in ("yolo_txt", "voc_xml"), f"{key}: 未知 label_format={fmt}")

    images = sorted(img_dir.glob(f"*{spec['image_ext']}"))
    need(len(images) > 0, f"{key}: 未找到 {spec['image_ext']} 图像，请核对 image_ext")
    mapping = spec.get("source_class_ids")

    if fmt == "voc_xml":
        # CUBIT-Det 官方分发 VOC XML（hk0000.xml 等），必须转成 YOLO txt 才能进 Ultralytics。
        if not mapping:
            xmls = sorted(lbl_dir.glob("*.xml"))[:200]
            print(f"[STOP] {key} 是 VOC XML，但 source_class_ids 未填。")
            print(f"       扫描前 {len(xmls)} 个 XML，实际类别名 = {inspect_voc_names(xmls)}")
            print("       请把这些名字映射到统一 ID 后填入 configs/datasets.yaml 再重跑。")
            raise SystemExit(2)
        yolo_dir = root / spec.get("yolo_label_dir", "labels")
        pairs = []
        for im in images:
            xml = lbl_dir / f"{im.stem}.xml"
            need(xml.exists(), f"{key}: 缺少 VOC 标注 {xml.name}")
            voc_to_yolo(xml, yolo_dir, mapping)
            pairs.append((im, yolo_dir / f"{im.stem}.txt"))
        print(f"[info] {key}: VOC XML 已转换为 YOLO txt -> {yolo_dir}")
        return images, {"root": root, "pairs": pairs, "mapping": mapping}

    # DamCrack 自带 YOLO txt：只做类别 ID 对齐，不做格式转换。
    need(bool(mapping), f"{key}: 请先填 source_class_ids（原始类别到统一 ID 的映射），不要靠猜")
    # Ultralytics 推标签路径时按 images/ -> labels/ 替换，标签必须与 images/ 同级。
    # 官方把标签放在 labels/Yolo/ 嵌套目录里，这里归位到 labels/，避免训练时找不到标签。
    std_label_dir = root / "labels"
    if lbl_dir.resolve() != std_label_dir.resolve():
        std_label_dir.mkdir(parents=True, exist_ok=True)
        for im in images:
            src = lbl_dir / f"{im.stem}.txt"
            need(src.exists(), f"{key}: 缺少标签 {src}")
            dst = std_label_dir / f"{im.stem}.txt"
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"[info] {key}: 标签已归位到 {std_label_dir}（Ultralytics 需 images/ 与 labels/ 同级）")
        lbl_dir = std_label_dir
    pairs = []
    for im in images:
        lb = lbl_dir / f"{im.stem}.txt"
        need(lb.exists(), f"{key}: 缺少标签 {lb.name}")
        pairs.append((im, lb))
    return images, {"root": root, "pairs": pairs, "mapping": mapping}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    collected = {}
    for key in ("cubit_det", "damcrack"):
        imgs, m = collect(cfg["datasets"][key], key)
        collected[key] = (imgs, m)
        cls_counts: dict[int, int] = defaultdict(int)
        for _, lb in m["pairs"]:
            for cid, n in read_yolo_label(lb):
                cls_counts[cid] += n
        print(f"[{key}] images={len(imgs)} raw_class_counts={dict(sorted(cls_counts.items()))}")

    dc = cfg["datasets"]["damcrack"]
    images, meta = collected["damcrack"]
    ratios = cfg["split"]["ratios"]
    need(abs(sum(ratios.values()) - 1.0) < 1e-6, "split.ratios 之和必须为 1")

    strategy = cfg["split"]["strategy"]
    if strategy == "parent_image":
        need(bool(dc.get("parent_id_field")),
             "strategy=parent_image 但未填 parent_id_field。请先核实 DamCrack 是否提供母图元数据；"
             "若没有，改 strategy=near_duplicate 并在骨架 §3.7 如实改写协议")
        groups: dict[str, list[Path]] = defaultdict(list)
        field = dc["parent_id_field"]
        for im in images:
            # 约定：从文件名按分隔符取母图 ID；实际正则请按数据集命名填写
            parent = im.stem.split(field)[0] if field in im.stem else im.stem
            groups[parent].append(im)
        buckets = list(groups.values())
    elif strategy == "near_duplicate":
        print("[warn] 近重复哈希是近似协议：同源但不重叠的切片哈希可能差异很大，不能保证完全杜绝泄漏")
        print("[warn] 需人工抽查分组，并在论文 §3.7 如实写明、§5.4 列为效度威胁")
        print("[info] 按 dHash 分组，复杂度 O(n^2)，大图集需耐心")
        buckets = group_near_duplicate(images, int(cfg["split"]["near_duplicate_hamming"]))
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "split_protocol.md").write_text(
            "# 数据划分协议记录（供论文 §3.7 / §5.4 引用）\n\n"
            f"- 策略：near_duplicate（dHash，汉明距离阈值 {cfg['split']['near_duplicate_hamming']}）\n"
            f"- 分组数：{len(buckets)}\n"
            "- 性质：**近似协议**。同源但不重叠的切片可能落入不同划分，无法保证完全杜绝泄漏。\n"
            "- 论文写法：§3.7 如实描述为近似划分；§5.4 把「无法完全排除同源切片潜在泄漏」列为效度威胁。\n"
            "- 改进：若有采集序列 / 时间戳 / GPS，优先按采集批次分组；并人工抽查若干组确认效果。\n"
            "- 禁止写法：「已彻底消除母图切片泄漏」「按母图级严格划分」。\n",
            encoding="utf-8",
        )
    elif strategy == "filename_window":
        # DamCrack 文件名为 A1.jpg … A4000.jpg 连续编号，VOC XML 无母图 ID / 切片坐标。
        # 按编号窗口分组是比感知哈希更可靠的「近似采集批次」划分：同源相邻切片不会跨 split。
        import re

        win = int(cfg["split"].get("filename_window", 20))
        need(win > 0, "split.filename_window 必须为正整数")

        def _num(p: Path) -> int:
            m = re.search(r"(\d+)$", p.stem)
            return int(m.group(1)) if m else 0

        ordered = sorted(images, key=_num)
        buckets = [ordered[i:i + win] for i in range(0, len(ordered), win)]
        print(f"[info] 按文件名连续编号每 {win} 张一组（近似采集批次），共 {len(buckets)} 组")
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "split_protocol.md").write_text(
            "# 数据划分协议记录（供论文 §3.7 / §5.4 引用）\n\n"
            f"- 策略：filename_window（按文件名连续编号每 {win} 张视为一个采集批次），共 {len(buckets)} 组。\n"
            "- 依据：DamCrack 文件名 A1.jpg … A4000.jpg 连续编号；VOC XML 中无母图 ID / 切片坐标元数据，"
            "相邻编号很可能来自同一母图或同一采集序列。\n"
            "- 性质：**近似协议**。窗口大小无法从数据集元数据确证，仍可能把同源切片分到不同划分。\n"
            "- 论文写法：§3.7 如实描述为「按采集序号的近似分组划分」；§5.4 把"
            "「无法完全排除同源切片潜在泄漏」列为效度威胁。\n"
            "- 建议：人工抽查若干连续编号图像确认内容是否连贯；若后续获得元数据，改用严格母图级划分。\n"
            "- 禁止写法：「严格母图级划分」「已彻底消除切片泄漏」。\n",
            encoding="utf-8",
        )
    else:
        raise SystemExit(f"[STOP] 未知 strategy: {strategy}")

    random.Random(cfg["seed"]).shuffle(buckets)
    n = len(buckets)
    n_tr = int(n * ratios["train"])
    n_va = int(n * ratios["val"])
    splits = {
        "train": [p for g in buckets[:n_tr] for p in g],
        "val": [p for g in buckets[n_tr:n_tr + n_va] for p in g],
        "test": [p for g in buckets[n_tr + n_va:] for p in g],
    }
    overlap = set(splits["train"]) & set(splits["test"])
    need(not overlap, f"划分泄漏：train/test 有 {len(overlap)} 张重叠图像")
    print(f"[split] groups={n} strategy={strategy} " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))

    if args.dry_run:
        print("[dry-run] 未写文件。确认统计无误后去掉 --dry-run")
        return

    out_dir = ROOT / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_files = {}
    for k, paths in splits.items():
        f = out_dir / f"damcrack_{k}.txt"
        f.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
        split_files[k] = f
        append_csv("split_summary.csv", {"dataset": "damcrack", "split": k,
                                        "images": len(paths), "strategy": strategy, "seed": cfg["seed"]})

    write_ultralytics_yaml(ROOT / "data" / "damcrack.yaml", meta["root"], split_files, cfg["classes"]["target"])
    cb_images, cb_meta = collected["cubit_det"]
    cb_shuffled = list(cb_images)
    random.Random(cfg["seed"]).shuffle(cb_shuffled)
    n_cb = len(cb_shuffled)
    n_cb_tr = int(n_cb * ratios["train"])
    n_cb_va = int(n_cb * ratios["val"])
    cb_splits = {
        "train": cb_shuffled[:n_cb_tr],
        "val": cb_shuffled[n_cb_tr:n_cb_tr + n_cb_va],
        "test": cb_shuffled[n_cb_tr + n_cb_va:],
    }
    cb_files = {}
    for k, paths in cb_splits.items():
        f = out_dir / f"cubit_det_{k}.txt"
        f.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
        cb_files[k] = f
        append_csv("split_summary.csv", {"dataset": "cubit_det", "split": k,
                                        "images": len(paths), "strategy": "random_image_level",
                                        "seed": cfg["seed"]})
    write_ultralytics_yaml(ROOT / "data" / "cubit_det.yaml", cb_meta["root"],
                           cb_files, cfg["classes"]["source"])

    (RESULTS / "class_protocol.md").write_text(
        "# 类别、格式与指标口径（供论文 §3.2 引用）\n\n"
        "- 源域 CUBIT-Det：3 类（crack / spalling / moisture），指标记为 mAP50^S。\n"
        "- 目标域 DamCrack：2 类（crack / spalling），指标记为 mAP50^T。\n"
        "- moisture 仅在源域测试集报告；两者 mAP 不可直接比较。\n"
        "- CUBIT-Det 官方分发 VOC XML，已由脚本转换为 YOLO txt。\n"
        "- DamCrack 自带 YOLO .txt，预处理只做类别 ID 与目录对齐，未做格式转换。\n",
        encoding="utf-8",
    )
    print(f"[check] CUBIT-Det 图像数 = {n_cb}；论文标称 5527，若不一致说明下载不完整或分享只含子集")
    print("[note] 源域按图像级随机划分；若发现采集序列 / 时间戳元数据，应改按批次分组以避免近重复泄漏")
    print("[done] 已生成 data/cubit_det.yaml 与 data/damcrack.yaml")


if __name__ == "__main__":
    main()
