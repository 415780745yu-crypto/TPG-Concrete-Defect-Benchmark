"""增量 ④：失败案例定性图（zero-shot vs 50-shot 同图对比 + FP/FN 着色）。

匹配阈值 IoU=0.5，画法：绿=真值、蓝=TP、红=FP、橙=FN。
为规避 6GB 显存 OOM：逐模型加载、只预测 --n 张、模型间清缓存。
产物：results/fig7_failure_cases.png、results/failure_case_stats.csv。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, RESULTS, load_cfg, need

IOU_THR = 0.5


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_yolo(txt, imgsz):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0])); cx, cy, w, h = (float(x) * imgsz for x in p[1:5])
        conf = float(p[5]) if len(p) >= 6 else 1.0
        boxes.append((cid, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
    return boxes


def match(gt_boxes, pred_boxes):
    used = [False] * len(gt_boxes)
    tp = fp = 0
    for p in sorted(pred_boxes, key=lambda x: -x[5]):
        best_i, best_v = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if not used[j]:
                v = _iou(p[1:5], g[1:5])
                if v > best_v:
                    best_v, best_i = v, j
        if best_i >= 0 and best_v >= IOU_THR:
            used[best_i] = True; tp += 1
        else:
            fp += 1
    fn = sum(1 for u in used if not u)
    return tp, fp, fn


def dump_predictions(model, images, pred_base, conf, imgsz):
    import torch
    model.predict(source=[str(p) for p in images], conf=conf, imgsz=imgsz,
                  save=False, save_txt=True, save_conf=True, project=str(pred_base),
                  name="labels", exist_ok=True, verbose=False)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-weights", required=True)
    ap.add_argument("--shot-weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    cfg = load_cfg()
    dc = cfg["datasets"]["damcrack"]
    root = Path(dc["root"]) if Path(dc["root"]).is_absolute() else ROOT / dc["root"]
    img_dir = root / dc["image_dir"]
    lbl_dir = root / dc["label_dir"]
    test_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    need(test_txt.exists(), "缺少 damcrack 测试划分")
    imgs = [Path(p) for p in test_txt.read_text(encoding="utf-8").splitlines() if p.strip()][: args.n]

    from ultralytics import YOLO
    import torch

    zero_pred_base = RESULTS / "predictions" / "zero_shot"
    dump_predictions(YOLO(str(Path(args.zero_weights))), imgs, zero_pred_base, args.conf, cfg["imgsz"])
    shot_pred_base = RESULTS / "predictions" / "shot50_seed42"
    dump_predictions(YOLO(str(Path(args.shot_weights))), imgs, shot_pred_base, args.conf, cfg["imgsz"])

    def pred_dir(base):
        p = base / "labels" / "labels"
        return p if p.exists() else base / "labels"

    zp = pred_dir(zero_pred_base)
    sp = pred_dir(shot_pred_base)

    try:
        import cv2
        import numpy as np
    except ImportError:
        raise SystemExit("[STOP] 需要 opencv-python")

    rows = []
    panels = []
    for im in imgs:
        g = parse_yolo(lbl_dir / f"{im.stem}.txt", cfg["imgsz"])
        z = parse_yolo(zp / f"{im.stem}.txt", cfg["imgsz"])
        s = parse_yolo(sp / f"{im.stem}.txt", cfg["imgsz"])
        tz, fz, nz = match(g, z)
        ts, fs, ns = match(g, s)
        rows.append({"image": im.name, "model": "zero_shot", "tp": tz, "fp": fz, "fn": nz})
        rows.append({"image": im.name, "model": "shot50", "tp": ts, "fp": fs, "fn": ns})

        img = cv2.imread(str(img_dir / im.name))
        if img is None:
            continue
        img = cv2.resize(img, (cfg["imgsz"], cfg["imgsz"]))
        def draw(frame, gt, pred):
            for cid, x1, y1, x2, y2, _ in gt:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)
            for cid, x1, y1, x2, y2, conf in pred:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 220), 1)
            return frame
        zf = draw(img.copy(), g, z)
        sf = draw(img.copy(), g, s)
        panels.append(np.hstack([zf, sf]))

    if panels:
        n_cols = 3
        n_rows = (len(panels) + n_cols - 1) // n_cols
        h, w = panels[0].shape[:2]
        canvas = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
        for i, p in enumerate(panels):
            r, c = divmod(i, n_cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
        cv2.imwrite(str(RESULTS / "fig7_failure_cases.png"), canvas)
        print(f"[ok] -> results/fig7_failure_cases.png ({len(panels)} 对图)")

    with open(RESULTS / "failure_case_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=["image", "model", "tp", "fp", "fn"])
        wcsv.writeheader(); wcsv.writerows(rows)
    print("[ok] -> results/failure_case_stats.csv")


if __name__ == "__main__":
    main()
