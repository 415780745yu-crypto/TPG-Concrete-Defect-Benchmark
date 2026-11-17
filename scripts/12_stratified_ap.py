"""增量 ③：逐尺度分层 AP（COCO 面积三档 + AP–√area 连续曲线）。

口径（论文 §4.4 / 用户确认）：
  - 面积在 640 评估尺度下计算：area = (w_norm*640) * (h_norm*640)。
  - 三档：small < 32²、medium 32²–96²、large ≥ 96²（px²）。
  - 某档某类真值实例 < 50 时，该档不单列，改用 8 等量（equal-instance-count）
    分箱的 AP–√area 连续曲线，禁止自定义边长分档。

用法（默认自行保存预测；也可复用 --pred-dir）：
  python scripts/12_stratified_ap.py --weights runs/pw_yolo_source_pretrain/weights/best.pt \
      --data data/damcrack.yaml --tag zero_shot
产物：results/stratified_ap.csv（长表）、results/tables/figS_stratified_area.csv、
      results/figS_stratified_area.png。
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import ROOT, RESULTS, append_csv, load_cfg, need, predict_to_txt

IMGSZ = 640
BINS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf"))]
N_AREA_BINS = 8
MIN_BIN_GT = 50
IOU_THR = 0.5


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(gt: list[tuple[int, tuple]], pred: list[tuple[int, tuple, float]]) -> float:
    """AP@0.5，all-point 插值。gt: [(img_id, box)]，pred: [(img_id, box, conf)]（已按 conf 降序）。"""
    if not gt:
        return 0.0
    gt_by_img: dict[int, list[tuple]] = {}
    for img_id, box in gt:
        gt_by_img.setdefault(img_id, []).append(box)
    matched: dict[int, list[bool]] = {k: [False] * len(v) for k, v in gt_by_img.items()}
    pred = sorted(pred, key=lambda x: -x[2])
    tp, fp = 0, 0
    n_gt = len(gt)
    recalls, precisions = [], []
    for img_id, box, _ in pred:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_by_img.get(img_id, [])):
            if matched[img_id][j]:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= IOU_THR:
            matched[img_id][best_j] = True
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    # 平滑后积分
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recalls, precisions):
        if r > prev_r:
            ap += (r - prev_r) * p
            prev_r = r
    return ap


def parse_yolo(txt: Path, imgsz: int):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(x) * imgsz for x in parts[1:5])
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        boxes.append((cid, x1, y1, x2, y2, conf))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pred-dir", default=None, help="已有预测目录（results/predictions/<tag>），跳过重新预测")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    cfg = load_cfg()
    dc = cfg["datasets"]["damcrack"]
    root = Path(dc["root"]) if Path(dc["root"]).is_absolute() else ROOT / dc["root"]
    lbl_dir = root / dc["label_dir"]
    test_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    need(test_txt.exists(), "缺少 damcrack 测试划分")
    test_images = [Path(p) for p in test_txt.read_text(encoding="utf-8").splitlines() if p.strip()]

    # 1) 预测（或复用）
    if args.pred_dir:
        pred_base = Path(args.pred_dir)
    else:
        pred_base = RESULTS / "predictions" / args.tag
    pred_labels = pred_base / "labels" / "labels"
    if not pred_labels.exists():
        pred_labels = pred_base / "labels"
    if not pred_labels.exists():
        from ultralytics import YOLO
        weights = Path(args.weights)
        weights = weights if weights.is_absolute() else ROOT / weights
        need(weights.exists(), f"权重不存在: {weights}")
        model = YOLO(str(weights))
        pred_labels = predict_to_txt(model, test_images, pred_base, args.conf, cfg["imgsz"])

    # 2) 读 GT 与预测
    names = cfg["classes"]["target"]  # {0: crack, 1: spalling}
    gt_by_class = {c: [] for c in names}
    pred_by_class = {c: [] for c in names}
    for img_id, im in enumerate(test_images):
        for (cid, x1, y1, x2, y2, conf) in parse_yolo(lbl_dir / f"{im.stem}.txt", IMGSZ):
            if cid in gt_by_class:
                gt_by_class[cid].append((img_id, (x1, y1, x2, y2)))
        for (cid, x1, y1, x2, y2, conf) in parse_yolo(pred_labels / f"{im.stem}.txt", IMGSZ):
            if cid in pred_by_class:
                pred_by_class[cid].append((img_id, (x1, y1, x2, y2), conf))

    # 3) 每类总体 AP50 + 三档 + 等量 √area 曲线
    for cid in names:
        cname = names[cid]
        gt_c = gt_by_class[cid]
        pred_c = pred_by_class[cid]
        overall = ap50(gt_c, pred_c)
        append_csv("stratified_ap.csv", {
            "tag": args.tag, "class": cname, "bin": "overall",
            "n_gt": len(gt_c), "ap50": round(overall, 4), "sqrt_area_bin": "",
        })
        for bname, lo, hi in BINS:
            g = [(i, b) for i, b in gt_c if lo <= (b[2] - b[0]) * (b[3] - b[1]) < hi]
            p = [(i, b, s) for i, b, s in pred_c if lo <= (b[2] - b[0]) * (b[3] - b[1]) < hi]
            apv = round(ap50(g, p), 4) if len(g) >= MIN_BIN_GT else ""
            append_csv("stratified_ap.csv", {
                "tag": args.tag, "class": cname, "bin": bname,
                "n_gt": len(g), "ap50": apv, "sqrt_area_bin": "",
            })

        # 等量分箱 AP–√area 曲线（8 箱，按 √area 排序后等实例数分箱）
        areas = sorted(((b[2] - b[0]) * (b[3] - b[1]) ** 0.5, i, b) for i, b in gt_c)
        n = len(areas)
        if n >= 8:
            k = n // N_AREA_BINS
            curve_rows = []
            for b in range(N_AREA_BINS):
                lo = b * k
                hi = n if b == N_AREA_BINS - 1 else (b + 1) * k
                chunk = areas[lo:hi]
                g = [(i, bb) for _, i, bb in chunk]
                center = sum(a for a, _, _ in chunk) / len(chunk)
                p = [(i, bb, s) for i, bb, s in pred_c if (bb[2] - bb[0]) * (bb[3] - bb[1]) ** 0.5
                     >= areas[lo][0] and (bb[2] - bb[0]) * (bb[3] - bb[1]) ** 0.5 <= areas[hi - 1][0]]
                curve_rows.append((round(center, 2), len(g), round(ap50(g, p), 4)))
            # 写曲线 CSV（每类一份）
            with open(RESULTS / "tables" / f"figS_stratified_area_{cname}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["sqrt_area", "n_gt", "ap50"])
                w.writerows(curve_rows)
            print(f"[ok] {args.tag}/{cname}: overall AP50={overall:.4f}, √area 曲线 {N_AREA_BINS} 点 -> "
                  f"results/tables/figS_stratified_area_{cname}.csv")

    print(f"[done] {args.tag} -> results/stratified_ap.csv")


if __name__ == "__main__":
    main()
