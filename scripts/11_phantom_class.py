"""增量 ②：零样本迁移的「幻影类」分析（moisture 假阳性）。

DamCrack 没有 moisture 类（目标域 2 类），但源域训练的 YOLOv8n 检测头是 3 类，
zero-shot 时仍会输出 class=2（moisture）的框——这些必然全是假阳性。
本脚本用 source_pretrain 权重在 DamCrack 测试集上做一次预测，统计幻影类的
检出数量、出现比例与置信度分布，产物 results/phantom_class.csv 与
results/fig7b_phantom_conf.png。零训练、只复用已有预测。

用法：
  python scripts/11_phantom_class.py \
      --weights runs/pw_yolo_source_pretrain/weights/best.pt \
      --data data/damcrack.yaml --tag zero_shot
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, RESULTS, append_csv, load_cfg, need, predict_to_txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", default="zero_shot")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    cfg = load_cfg()
    dc = cfg["datasets"]["damcrack"]
    root = Path(dc["root"]) if Path(dc["root"]).is_absolute() else ROOT / dc["root"]
    lbl_dir = root / dc["label_dir"]
    test_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    need(test_txt.exists(), "缺少 damcrack 测试划分，先跑 01_prepare_data.py")

    from ultralytics import YOLO
    weights = Path(args.weights)
    weights = weights if weights.is_absolute() else ROOT / weights
    need(weights.exists(), f"权重不存在: {weights}")

    # 1) 统计 GT：目标域只有 0/1 两类，class2 应为 0（用于佐证「幻影」）
    gt_by_class = {0: 0, 1: 0, 2: 0}
    test_images = [Path(p) for p in test_txt.read_text(encoding="utf-8").splitlines() if p.strip()]
    for im in test_images:
        lb = lbl_dir / f"{im.stem}.txt"
        if lb.exists():
            for line in lb.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cid = int(float(line.split()[0]))
                    gt_by_class[cid] = gt_by_class.get(cid, 0) + 1
    need(gt_by_class.get(2, 0) == 0, "DamCrack 测试集出现 class2 真值，类别映射有误，请核对")

    # 2) 预测并保存逐图结果（保存 conf，便于分布统计）
    out_dir = RESULTS / "predictions" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    pred_dir = predict_to_txt(model, test_images, out_dir, args.conf, cfg["imgsz"])

    # 3) 解析保存的预测文本
    det_by_class = {0: 0, 1: 0, 2: 0}
    confs_class2 = []
    n_imgs_class2 = 0
    confs_class2_ge025 = []
    n_imgs_class2_ge025 = 0
    n_class2_ge05 = 0
    for im in test_images:
        txt = pred_dir / f"{im.stem}.txt"
        boxes = []
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 6:
                    cid = int(float(parts[0])); conf = float(parts[5])
                    boxes.append((cid, conf))
        has2 = False
        has2_ge025 = False
        for cid, conf in boxes:
            det_by_class[cid] = det_by_class.get(cid, 0) + 1
            if cid == 2:
                has2 = True
                confs_class2.append(conf)
                if conf >= 0.25:
                    has2_ge025 = True
                    confs_class2_ge025.append(conf)
                if conf >= 0.5:
                    n_class2_ge05 += 1
        if has2:
            n_imgs_class2 += 1
        if has2_ge025:
            n_imgs_class2_ge025 += 1

    total_imgs = len(test_images)
    phantom = det_by_class.get(2, 0)
    row1 = {
        "tag": args.tag, "split": "test", "weights": str(weights),
        "n_images": total_imgs, "gt_class0": gt_by_class[0], "gt_class1": gt_by_class[1],
        "det_class0": det_by_class.get(0, 0), "det_class1": det_by_class.get(1, 0),
        "det_class2_phantom": phantom,
        "images_with_phantom": n_imgs_class2,
        "phantom_rate_per_image": round(n_imgs_class2 / total_imgs, 4) if total_imgs else 0,
        "phantom_mean_conf": round(sum(confs_class2) / len(confs_class2), 4) if confs_class2 else "",
        "phantom_max_conf": round(max(confs_class2), 4) if confs_class2 else "",
        "conf_threshold": args.conf,
        "n_phantom_ge_05": n_class2_ge05,
        "note": "conf=0.001 是 AP 评估口径（全阈值统计）",
    }
    append_csv("phantom_class.csv", row1)
    row2 = {
        "tag": args.tag, "split": "test", "weights": str(weights),
        "n_images": total_imgs, "gt_class0": gt_by_class[0], "gt_class1": gt_by_class[1],
        "det_class0": "", "det_class1": "",
        "det_class2_phantom": len(confs_class2_ge025),
        "images_with_phantom": n_imgs_class2_ge025,
        "phantom_rate_per_image": round(n_imgs_class2_ge025 / total_imgs, 4) if total_imgs else 0,
        "phantom_mean_conf": round(sum(confs_class2_ge025) / len(confs_class2_ge025), 4) if confs_class2_ge025 else "",
        "phantom_max_conf": round(max(confs_class2_ge025), 4) if confs_class2_ge025 else "",
        "conf_threshold": 0.25,
        "n_phantom_ge_05": n_class2_ge05,
        "note": "conf>=0.25 部署误检量级参考；conf>=0.5 框数见 n_phantom_ge_05",
    }
    append_csv("phantom_class.csv", row2)
    print(f"[ok] phantom(moisture) 检出 {phantom} 个框 / {n_imgs_class2} 张图（共 {total_imgs}），"
          f"mean_conf={row1['phantom_mean_conf']}；conf>=0.25 共 {len(confs_class2_ge025)} 框 / "
          f"{n_imgs_class2_ge025} 图，>=0.5 共 {n_class2_ge05} 框 -> results/phantom_class.csv")

    # 4) 幻影类置信度直方图（若 matplotlib 可用）
    if confs_class2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.hist(confs_class2, bins=50, range=(0, 1), color="#c0392b", alpha=0.85)
            ax.set_xlabel("confidence of phantom 'moisture' detections")
            ax.set_ylabel("count")
            ax.set_title(f"zero-shot phantom class (moisture), {args.tag}")
            fig.tight_layout()
            fig.savefig(RESULTS / "fig7b_phantom_conf.png", dpi=150)
            print(f"[ok] -> results/fig7b_phantom_conf.png")
        except Exception as exc:
            print(f"[warn] 直方图未生成: {exc}")


if __name__ == "__main__":
    main()
