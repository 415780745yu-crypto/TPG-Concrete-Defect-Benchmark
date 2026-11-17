"""阶段 5：端侧实测（骨架 §4.5 / 表 6 上半 / 图 8）。

用 Ultralytics 直接加载 ncnn 模型目录做 predict 计时，避免自己写后处理带来的口径偏差。
记录 batch=1、预热轮数、线程数、重复次数；禁止由 GFLOPs 推算 FPS。
功耗需外接 USB 功率计，本脚本不伪造该列；没有设备就在表 6 删除功耗列。

用法：
  python scripts/06_benchmark.py --model runs/.../best_ncnn_model --platform x86_cpu --precision fp32 --tag pw_yolo_x86_fp32
  python scripts/06_benchmark.py --model runs/.../best_ncnn_model --platform rpi5 --precision int8 --tag pw_yolo_rpi5_int8
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from common import ROOT, append_csv, load_cfg, need


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ncnn 模型目录或 .pt 权重")
    ap.add_argument("--platform", required=True, choices=["x86_cpu", "rpi4b", "rpi5"])
    ap.add_argument("--precision", required=True, choices=["fp32", "int8"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", default=None, help="单张图或图像目录；不填则生成一张固定噪声图")
    ap.add_argument("--map50", default="", help="该行模型在评估集上的 mAP50（FP32 行填 base mAP^S，INT8 行填量化后 mAP）")
    ap.add_argument("--cpu-model", default="Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz")
    ap.add_argument("--quantized-map50", default="", help="INT8 必填：量化后 mAP 回测值")
    args = ap.parse_args()

    cfg = load_cfg()
    dep = cfg["deploy"]
    mp = Path(args.model)
    mp = mp if mp.is_absolute() else ROOT / mp
    need(mp.exists(), f"模型路径不存在: {mp}")
    need(args.precision != "int8" or args.quantized_map50,
         "INT8 必须同时记录量化后 mAP 回测值（--quantized-map50），否则表 6 不完整")
    if args.precision == "int8":
        need(int(dep["int8_calib_images"]) > 0,
             "请在 configs/datasets.yaml 填 int8_calib_images（校准集张数）")

    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        raise SystemExit("[STOP] 请先 pip install -r requirements.txt")

    model = YOLO(str(mp), task="detect")
    if args.source:
        src = args.source
    else:
        img = (np.random.rand(dep["imgsz"], dep["imgsz"], 3) * 255).astype("uint8")
        src = img

    for _ in range(dep["warmup"]):
        model.predict(src, imgsz=dep["imgsz"], verbose=False)
    lat = []
    for _ in range(dep["repeats"]):
        t0 = time.perf_counter()
        model.predict(src, imgsz=dep["imgsz"], verbose=False)
        lat.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(lat)
    fps = 1000.0 / mean_ms
    size_mb = ""
    if mp.is_file():
        size_mb = round(mp.stat().st_size / 1e6, 3)
    elif mp.is_dir():
        size_mb = round(sum(f.stat().st_size for f in mp.rglob("*") if f.is_file()) / 1e6, 3)

    row = {
        "tag": args.tag, "platform": args.platform, "precision": args.precision,
        "model": str(mp), "size_MB": size_mb, "latency_ms_mean": round(mean_ms, 3),
        "latency_ms_std": round(statistics.pstdev(lat), 3), "fps": round(fps, 2),
        "warmup": dep["warmup"], "repeats": dep["repeats"], "threads": dep["threads"],
        "batch": 1, "imgsz": dep["imgsz"],
        "map50": args.map50,
        "source": args.source if args.source else "noise(640x640 random)",
        "cpu_model": args.cpu_model,
        "quantized_map50": args.quantized_map50,
        "power_W": "",
    }
    append_csv("deploy.csv", row)
    print(f"[ok] {args.tag}: {mean_ms:.2f} ms / {fps:.2f} FPS -> results/deploy.csv")
    print("[提醒] power_W 留空：整机功耗需外接功率计实测，无设备则在表 6 删除该列，不要写「实测」")


if __name__ == "__main__":
    main()
