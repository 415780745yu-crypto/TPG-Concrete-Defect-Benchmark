"""阶段 2–4：统一评估，输出 mAP50 / mAP50-95 / 分类别 AP / 参数量 / GFLOPs。

用法：
  python scripts/04_eval.py --weights <best.pt> --data data/damcrack.yaml --tag zero_shot --protocol crossdomain
  python scripts/04_eval.py --weights <best.pt> --data data/damcrack.yaml --tag a2_pconv --protocol ablation

--protocol 决定结果写入哪张表的数据源：crossdomain -> crossdomain.csv；ablation -> ablation.csv；
其他 -> eval_log.csv。所有数值均为实测，脚本不做任何估算。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, append_csv, load_cfg, need


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--protocol", default="", help="crossdomain | ablation | (空)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = load_cfg()
    w, d = Path(args.weights), Path(args.data)
    w = w if w.is_absolute() else ROOT / w
    d = d if d.is_absolute() else ROOT / d
    need(w.exists(), f"权重不存在: {w}")
    need(d.exists(), f"data.yaml 不存在: {d}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 请先 pip install -r requirements.txt")

    model = YOLO(str(w))
    metrics = model.val(data=str(d), split=args.split, imgsz=cfg["imgsz"], verbose=False)
    box = metrics.box

    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = ""
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = round(get_flops(model.model, cfg["imgsz"]), 3)
    except Exception as exc:
        print(f"[warn] GFLOPs 未取得，留空待补: {exc}")

    names = metrics.names
    ap50_by_cls = {}
    try:
        for i, name in names.items():
            ap50_by_cls[name] = round(float(box.ap50[i]), 4)
    except Exception as exc:
        print(f"[warn] 分类别 AP 未取得: {exc}")

    row = {
        "tag": args.tag, "protocol": args.protocol, "weights": str(w), "data": str(d),
        "split": args.split, "map50": round(float(box.map50), 4),
        "map5095": round(float(box.map), 4),
        "params_M": round(params_m, 4), "gflops": gflops, "imgsz": cfg["imgsz"],
        "note": args.note,
    }
    for k, v in ap50_by_cls.items():
        row[f"ap50_{k}"] = v

    target = {"crossdomain": "crossdomain.csv", "ablation": "ablation.csv"}.get(args.protocol, "eval_log.csv")
    append_csv(target, row)
    if target != "eval_log.csv":
        append_csv("eval_log.csv", row)
    print(f"[ok] {args.tag} mAP50={row['map50']} params={row['params_M']}M -> results/{target}")


if __name__ == "__main__":
    main()
