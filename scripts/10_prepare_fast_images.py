"""阶段 1.5：把 CUBIT-Det 原图预缩放到标准布局 images/，降低训练 I/O。

目录约定（重要）
--------------
- images_full/  : 4624×3472 原图，17.8 GB，只作存档与分析来源，不参与训练
- images/       : 640×480 近似等比预缩放训练副本，651 MB（本脚本产物）
- labels/       : YOLO txt，由 01_prepare_data.py 从 Annotations/ 转换
- Annotations/  : 原始 VOC XML

**images/ 这个名字不能改**：Ultralytics 按 images→labels 推标签路径。
曾把预缩放目录命名为 images_640，结果训练时 "no labels found"、box_loss=0，
指标全 0。若改目录名，必须同步把标签放到图像同目录。

背景
----
原图单张全分辨率解码约 173 ms，是训练主要瓶颈；模型本就在 imgsz=640 下训练，
高分辨率细节在数据加载时已被丢弃。本脚本把原图近似等比（4:3）预缩放到
640×480 并以 JPEG q95 重新编码，标签为归一化坐标，**无需任何改动**。

安全性
------
- 只写入派生目录 images/，不修改、不删除 images_full/ 下任何原图。
- 处理前先做 20 张等价性自检：比较「原图全分辨率 → INTER_LINEAR 640×480」
  与「本脚本实际管线（REDUCED_4 → INTER_AREA 640×480 → JPEG q95）」的像素差，
  打印 MAE / PSNR；若 PSNR < 35 dB 会停止，避免静默引入明显失真。
- Windows 中文路径统一用 np.fromfile + imdecode / imencode + tofile。

用法
----
  C:/Users/zhenbai/anaconda3/python.exe scripts/10_prepare_fast_images.py
  （可重复运行，已生成的文件会跳过）

configs/datasets.yaml 的 cubit_det.image_dir 固定为 images（当前已如此），
重跑 01_prepare_data.py 即可生成新的 split txt 与 data.yaml。
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\YaLi\CUBIT-Det")
SRC = ROOT / "images_full"
DST = ROOT / "images"
SIZE = (640, 480)
QUALITY = 95


def _read(path: Path, flag: int):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise RuntimeError("empty file")
    im = cv2.imdecode(buf, flag)
    if im is None:
        raise RuntimeError("imdecode failed")
    return im


def equivalence_check(files: list[Path], n: int = 20) -> None:
    maes, psnrs = [], []
    for p in files[:n]:
        ref = cv2.resize(_read(p, cv2.IMREAD_COLOR), SIZE, interpolation=cv2.INTER_LINEAR)
        reduced = cv2.resize(_read(p, cv2.IMREAD_REDUCED_COLOR_4), SIZE,
                             interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", reduced, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
        if not ok:
            raise RuntimeError("imencode failed")
        pre = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        d = np.abs(ref.astype(np.float32) - pre.astype(np.float32))
        maes.append(float(np.mean(d)))
        psnrs.append(float(10 * np.log10(255 ** 2 / (np.mean(d ** 2) + 1e-9))))
    mae, psnr = float(np.mean(maes)), float(np.mean(psnrs))
    print(f"[check] n={len(maes)} | MAE={mae:.4f}/255 | PSNR={psnr:.2f} dB", flush=True)
    if psnr < 35:
        raise SystemExit("[STOP] 实际管线与原图下采样差异过大，请先检查插值/编码设置")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"[STOP] 原图目录不存在: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.jpg"))
    print(f"source images: {len(files)} | target size: {SIZE} | jpeg q{QUALITY}", flush=True)
    if not files:
        raise SystemExit("[STOP] 原图目录为空")
    equivalence_check(files)

    t0 = time.perf_counter()
    done = skipped = failed = 0
    for i, p in enumerate(files, 1):
        out = DST / p.name
        if out.exists():
            skipped += 1
            continue
        try:
            reduced = cv2.resize(_read(p, cv2.IMREAD_REDUCED_COLOR_4), SIZE,
                                 interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", reduced,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
            if not ok:
                raise RuntimeError("imencode failed")
            enc.tofile(str(out))
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  FAIL {p.name}: {exc}", flush=True)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} | elapsed {time.perf_counter() - t0:.0f}s", flush=True)

    outs = list(DST.glob("*.jpg"))
    total_mb = sum(f.stat().st_size for f in outs) / 1e6
    print(f"done={done} skipped={skipped} failed={failed}", flush=True)
    print(f"output files={len(outs)} | size={total_mb:.1f} MB | "
          f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)
    if failed:
        print("[warn] 有失败文件，重跑本脚本会补上（已生成的会跳过）", flush=True)
    if len(outs) != len(files):
        print("[warn] 输出数量与原图不一致，请核对后再改配置", flush=True)


if __name__ == "__main__":
    main()
