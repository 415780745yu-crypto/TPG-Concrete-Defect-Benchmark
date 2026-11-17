"""增量 ⑤：域距离–性能逐图 Spearman 相关（光度 + 频域双通道）。

口径（用户确认）：
  - 光度统计距离 = 亮度(mean) + 对比度(std) + 直方图(64-bin, 1-交集) 的归一化组合，
    各自 [0,1]，再取均值作为「光度距离」；不用光流（静态图无光流）。
  - 频域距离 = 平均径向功率谱（归一化到单位和的 log 曲线）与源域均值谱的 0.5*L1。
  - 逐图指标：IoU>=0.5 计 TP；recall = TP/|GT|。空标签负样本图 recall 未定义 →
    主相关排除负样本并注明；另对全体图算 precision 做稳健性对照。
  - 各通道与逐图 recall 做 Spearman。

用法：
  python scripts/15_domain_perf_corr.py \
      --weights runs/pw_yolo_source_pretrain/weights/best.pt \
      --pred-dir results/predictions/zero_shot --tag zero_shot

产物：results/domain_perf_corr.csv、results/tables/fig_domain_perf_corr.csv、
      results/fig_domain_perf_corr.png
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from common import ROOT, RESULTS, load_cfg, need, predict_to_txt

IMGSZ = 640
IOU_THR = 0.5


def gray_stats(img_bgr):
    g = (img_bgr.mean(axis=2) if img_bgr.ndim == 3 else img_bgr).astype(np.float32)
    return g.mean() / 255.0, g.std() / 128.0


def gray_hist(img_bgr, bins=64):
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([g], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    h /= (h.sum() + 1e-9)
    return h


def radial_spectrum(img_bgr):
    g = (img_bgr.mean(axis=2) if img_bgr.ndim == 3 else img_bgr).astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h / 2, w / 2
    ys, xs = np.indices((h, w))
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(int)
    rmax = int(min(cy, cx)) - 1
    prof = np.zeros(rmax, dtype=np.float64)
    cnt = np.zeros(rmax, dtype=np.float64)
    for rr in range(rmax):
        mask = r == rr
        prof[rr] = mag[mask].sum()
        cnt[rr] = mask.sum()
    prof = prof / (cnt + 1e-9)
    prof = np.log1p(prof)
    prof /= (prof.sum() + 1e-9)
    return prof


def spearman(a, b):
    from scipy.stats import spearmanr
    def tof(v):
        try:
            f = float(v)
            return f if math.isfinite(f) else float("nan")
        except (TypeError, ValueError):
            return float("nan")
    a = np.asarray([tof(x) for x in a], dtype=float)
    b = np.asarray([tof(x) for x in b], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p = spearmanr(a[mask], b[mask])
    return float(rho), float(p), int(mask.sum())


def parse_yolo(txt, imgsz, with_conf=True):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0])); cx, cy, w, h = (float(x) * imgsz for x in p[1:5])
        conf = float(p[5]) if with_conf and len(p) >= 6 else 1.0
        boxes.append((cid, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", default="zero_shot")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--source-sample", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    src_cfg = cfg["datasets"]["cubit_det"]
    src_root = Path(src_cfg["root"]) if Path(src_cfg["root"]).is_absolute() else ROOT / src_cfg["root"]
    src_img_dir = src_root / src_cfg["image_dir"]
    dc = cfg["datasets"]["damcrack"]
    tgt_root = Path(dc["root"]) if Path(dc["root"]).is_absolute() else ROOT / dc["root"]
    tgt_img_dir = tgt_root / dc["image_dir"]
    tgt_lbl_dir = tgt_root / dc["label_dir"]

    src_train = ROOT / "data" / "splits" / "cubit_det_train.txt"
    tgt_test = ROOT / "data" / "splits" / "damcrack_test.txt"
    need(src_train.exists() and tgt_test.exists(), "缺少划分文件，先跑 01_prepare_data.py")

    n_sample = args.source_sample or int(cfg.get("domain_gap", {}).get("stat_sample", 200))
    src_imgs = [Path(p) for p in src_train.read_text(encoding="utf-8").splitlines() if p.strip()]
    step = max(1, len(src_imgs) // n_sample)
    src_sample = src_imgs[::step][:n_sample]

    # 1) 源域参考（光度统计 + 平均径向谱）
    import cv2
    b_means, b_stds, hists, specs = [], [], [], []
    for im in src_sample:
        img = cv2.imread(str(src_img_dir / im.name))
        if img is None:
            continue
        img = cv2.resize(img, (IMGSZ, IMGSZ))
        bm, bs = gray_stats(img)
        b_means.append(bm); b_stds.append(bs); hists.append(gray_hist(img))
        specs.append(radial_spectrum(img))
    src_mean_b = float(np.mean(b_means)); src_std_b = float(np.std(b_means)) + 1e-6
    src_mean_c = float(np.mean(b_stds)); src_std_c = float(np.std(b_stds)) + 1e-6
    src_hist = np.mean(np.stack(hists), axis=0)
    src_spec = np.mean(np.stack(specs), axis=0)

    # 2) 目标域逐图：距离 + 逐图 TP/FP/FN
    pred_base = Path(args.pred_dir) if args.pred_dir else RESULTS / "predictions" / args.tag
    pred_labels = pred_base / "labels" / "labels"
    if not pred_labels.exists():
        pred_labels = pred_base / "labels"
    if not pred_labels.exists():
        from ultralytics import YOLO
        weights = Path(args.weights)
        weights = weights if weights.is_absolute() else ROOT / weights
        need(weights.exists(), f"权重不存在: {weights}")
        tgt_imgs = [Path(p) for p in tgt_test.read_text(encoding="utf-8").splitlines() if p.strip()]
        pred_labels = predict_to_txt(YOLO(str(weights)), tgt_imgs, pred_base, args.conf, cfg["imgsz"])

    rows = []
    for im in [Path(p) for p in tgt_test.read_text(encoding="utf-8").splitlines() if p.strip()]:
        img = cv2.imread(str(tgt_img_dir / im.name))
        if img is None:
            continue
        img = cv2.resize(img, (IMGSZ, IMGSZ))
        bm, bs = gray_stats(img)
        bdist = abs(bm - src_mean_b) / src_std_b
        cdist = abs(bs - src_mean_c) / src_std_c
        h = gray_hist(img)
        hdist = 1.0 - float(np.minimum(h, src_hist).sum())
        photometric = (min(bdist, 1.0) + min(cdist, 1.0) + hdist) / 3.0
        sp = radial_spectrum(img)
        spectral = float(0.5 * np.abs(sp - src_spec).sum())

        gt = parse_yolo(tgt_lbl_dir / f"{im.stem}.txt", IMGSZ, with_conf=False)
        pred = parse_yolo(pred_labels / f"{im.stem}.txt", IMGSZ, with_conf=True)
        used = [False] * len(gt)
        tp = fp = 0
        for p in sorted(pred, key=lambda x: -x[5]):
            best_i, best_v = -1, 0.0
            for j, g in enumerate(gt):
                if not used[j]:
                    v = iou(p[1:5], g[1:5])
                    if v > best_v:
                        best_v, best_i = v, j
            if best_i >= 0 and best_v >= IOU_THR:
                used[best_i] = True; tp += 1
            else:
                fp += 1
        fn = sum(1 for u in used if not u)
        n_gt = len(gt)
        recall = tp / n_gt if n_gt > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rows.append(dict(image=im.name, brightness_dist=round(bdist, 4), contrast_dist=round(cdist, 4),
                         hist_dist=round(hdist, 4), photometric_dist=round(photometric, 4),
                         spectral_dist=round(spectral, 4), n_gt=n_gt, tp=tp, fp=fp, fn=fn,
                         recall=round(recall, 4) if math.isfinite(recall) else "",
                         precision=round(precision, 4) if math.isfinite(precision) else ""))

    with open(RESULTS / "domain_perf_corr.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # 3) Spearman：光度距离 vs recall、频域距离 vs recall；负样本排除并注明；precision 对照
    rho_b, p_b, n_b = spearman([r["photometric_dist"] for r in rows], [r["recall"] for r in rows])
    rho_s, p_s, n_s = spearman([r["spectral_dist"] for r in rows], [r["recall"] for r in rows])
    rho_p, p_p, n_p = spearman([r["photometric_dist"] for r in rows], [r["precision"] for r in rows])
    rho_s2, p_s2, n_s2 = spearman([r["spectral_dist"] for r in rows], [r["precision"] for r in rows])
    summary = [
        {"channel": "photometric", "metric": "recall", "spearman_rho": round(rho_b, 4), "p": round(p_b, 4), "n": n_b, "note": "excludes empty-label images"},
        {"channel": "spectral", "metric": "recall", "spearman_rho": round(rho_s, 4), "p": round(p_s, 4), "n": n_s, "note": "excludes empty-label images"},
        {"channel": "photometric", "metric": "precision", "spearman_rho": round(rho_p, 4), "p": round(p_p, 4), "n": n_p, "note": "includes empty-label images where defined"},
        {"channel": "spectral", "metric": "precision", "spearman_rho": round(rho_s2, 4), "p": round(p_s2, 4), "n": n_s2, "note": "includes empty-label images where defined"},
    ]
    with open(RESULTS / "tables" / "fig_domain_perf_corr.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print("[ok] Spearman -> results/tables/fig_domain_perf_corr.csv")
    for s in summary:
        print(f"    {s['channel']:>12} x {s['metric']:<10} rho={s['spearman_rho']} p={s['p']} n={s['n']}")

    # 4) 散点图（光度/频域 距离 vs recall）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rec = [r["recall"] if str(r["recall"]).strip() not in ("",) else float("nan") for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].scatter([r["photometric_dist"] for r in rows], rec, s=8, alpha=0.5)
        ax[0].set_xlabel("photometric distance"); ax[0].set_ylabel("per-image recall"); ax[0].set_title(f"rho={rho_b:.3f} (p={p_b:.3g})")
        ax[1].scatter([r["spectral_dist"] for r in rows], rec, s=8, alpha=0.5)
        ax[1].set_xlabel("spectral distance"); ax[1].set_ylabel("per-image recall"); ax[1].set_title(f"rho={rho_s:.3f} (p={p_s:.3g})")
        fig.tight_layout(); fig.savefig(RESULTS / "fig_domain_perf_corr.png", dpi=150)
        print("[ok] -> results/fig_domain_perf_corr.png")
    except Exception as exc:
        print(f"[warn] 散点图未生成: {exc}")


if __name__ == "__main__":
    main()
