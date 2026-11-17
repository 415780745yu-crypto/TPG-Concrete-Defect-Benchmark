"""阶段 5：导出 ONNX / NCNN（骨架 §3.6 / 表 6 下半）。

优先使用 Ultralytics 原生 ncnn 导出（内部调用 pnnx），失败再退回 ONNX + pnnx 命令行。
dynamic 必须关闭：PNNX 需要静态 shape。

用法：
  python scripts/05_export.py --weights <best.pt> --format ncnn --tag pw_yolo
  python scripts/05_export.py --weights <best.pt> --format onnx --tag pw_yolo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, append_csv, load_cfg, need


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", choices=["ncnn", "onnx"], default="ncnn")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-label", default="", help="对照实验标签，如 dsconv_yolo；导出失败也要如实记录")
    args = ap.parse_args()

    cfg = load_cfg()
    dep = cfg["deploy"]
    w = Path(args.weights)
    w = w if w.is_absolute() else ROOT / w
    need(w.exists(), f"权重不存在: {w}")
    need(dep["dynamic"] is False, "deploy.dynamic 必须为 false")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 请先 pip install -r requirements.txt")

    model = YOLO(str(w))
    status, artifact, log = "PASS", "", ""
    try:
        out = model.export(
            format=args.format,
            imgsz=dep["imgsz"],
            dynamic=False,
            simplify=dep["simplify"] if args.format == "onnx" else False,
        )
        artifact = str(out)
        print(f"[ok] 导出成功: {artifact}")
    except Exception as exc:
        status = "FAIL"
        log = f"{type(exc).__name__}: {exc}"
        print(f"[fail] 导出失败，原样记录日志（不要编造报错名）:\n{log}")

    if args.format == "onnx" and status == "PASS":
        print(f"\n[下一步] pnnx {artifact} inputshape=[1,3,{dep['imgsz']},{dep['imgsz']}]")
        print("[提示] 若 pnnx 不是 pip 包，请从 https://github.com/pnnx/pnnx 取预编译二进制")

    append_csv("export_log.csv", {
        "tag": args.tag, "model_label": args.model_label, "weights": str(w),
        "format": args.format, "imgsz": dep["imgsz"], "dynamic": False,
        "status": status, "artifact": artifact, "log": log,
    })
    print("[提醒] 表 6 下半的「不可导出」必须是真的跑过一次；只有日志没有 mAP 不足以构成精度-可部署性权衡")


if __name__ == "__main__":
    main()
