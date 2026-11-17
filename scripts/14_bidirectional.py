"""增量 ①：双向零样本（T→S）评估。

target_only 模型是 2 类（crack/spalling），源域 CUBIT-Det 是 3 类（多 moisture）。
T→S 只应在两域共享的 2 类上评估：忽略源域 moisture 真值，忽略模型输出里不可能存在的 class2。
匹配 IoU=0.5，AP50 用 all-point 插值；mAP50 = 两类 AP50 均值（与 Ultralytics 口径一致）。

用法：
  python scripts/14_bidirectional.py \
      --weights runs/pw_yolo_target_only/weights/best.pt --tag target_to_source
产物：追加一行到 results/crossdomain.csv。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, RESULTS, append_csv, load_cfg, need, predict_to_txt

IOU_THR = 0.5
SHARED = {0, 1}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(gt, pred):
    if not gt:
        return 0.0
    gt_by_img = {}
    for img_id, box in gt:
        gt_by_img.setdefault(img_id, []).append(box)
    matched = {k: [False] * len(v) for k, v in gt_by_img.items()}
    pred = sorted(pred, key=lambda x: -x[2])
    tp = fp = 0
    n_gt = len(gt)
    recalls, precisions = [], []
    for img_id, box, _ in pred:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_by_img.get(img_id, [])):
            if matched[img_id][j]:
                continue
            v = _iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= IOU_THR:
            matched[img_id][best_j] = True
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recalls, precisions):
        if r > prev_r:
            ap += (r - prev_r) * p
            prev_r = r
    return ap


def parse_yolo(txt, imgsz, with_conf=True):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0]))
        if cid not in SHARED:
            continue
        cx, cy, w, h = (float(x) * imgsz for x in p[1:5])
        conf = float(p[5]) if with_conf and len(p) >= 6 else 1.0
        boxes.append((cid, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", default="target_to_source")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    cfg = load_cfg()
    src_cfg = cfg["datasets"]["cubit_det"]
    src_root = Path(src_cfg["root"]) if Path(src_cfg["root"]).is_absolute() else ROOT / src_cfg["root"]
    img_dir = src_root / src_cfg["image_dir"]
    lbl_dir = src_root / src_cfg["yolo_label_dir"]
    test_txt = ROOT / "data" / "splits" / "cubit_det_test.txt"
    need(test_txt.exists(), "缺少 cubit_det 测试划分")
    imgs = [Path(p) for p in test_txt.read_text(encoding="utf-8").splitlines() if p.strip()]

    from ultralytics import YOLO
    import torch
    weights = Path(args.weights)
    weights = weights if weights.is_absolute() else ROOT / weights
    need(weights.exists(), f"权重不存在: {weights}")
    pred_base = RESULTS / "predictions" / args.tag
    model = YOLO(str(weights))
    pd = predict_to_txt(model, imgs, pred_base, args.conf, cfg["imgsz"])
    del model
    torch.cuda.empty_cache()

    gt = {0: [], 1: []}
    pred = {0: [], 1: []}
    for img_id, im in enumerate(imgs):
        for (cid, x1, y1, x2, y2, conf) in parse_yolo(lbl_dir / f"{im.stem}.txt", cfg["imgsz"], with_conf=False):
            gt[cid].append((img_id, (x1, y1, x2, y2)))
        for (cid, x1, y1, x2, y2, conf) in parse_yolo(pd / f"{im.stem}.txt", cfg["imgsz"], with_conf=True):
            pred[cid].append((img_id, (x1, y1, x2, y2), conf))

    ap_crack = ap50(gt[0], pred[0])
    ap_spalling = ap50(gt[1], pred[1])
    map50 = round((ap_crack + ap_spalling) / 2, 4)

    params_m = round(sum(p.numel() for p in YOLO(str(weights)).model.parameters()) / 1e6, 4)
    gflops = ""
    try:
        from ultralytics.utils.torch_utils import get_flops
        m = YOLO(str(weights)).model
        gflops = round(get_flops(m, cfg["imgsz"]), 3)
    except Exception as exc:
        print(f"[warn] GFLOPs 未取得: {exc}")

    row = {
        "tag": args.tag, "protocol": "bidirectional", "weights": str(weights),
        "data": str(ROOT / "data" / "cubit_det.yaml"), "split": "test",
        "map50": map50, "map5095": "", "params_M": params_m, "gflops": gflops,
        "imgsz": cfg["imgsz"],
        "note": "自有 AP 实现、仅 crack/spalling 两个共有类",
        "ap50_crack": round(ap_crack, 4), "ap50_spalling": round(ap_spalling, 4),
    }
    append_csv("crossdomain.csv", row)
    print(f"[ok] {args.tag} mAP50(T->S, 2共享类)={map50} (crack={round(ap_crack,4)}, spalling={round(ap_spalling,4)}) -> results/crossdomain.csv")


if __name__ == "__main__":
    main()
