"""22: INT8 量化（ncnn2table → ncnn2int8 → 组装目录）。只量化、不测速、不回测。

用法：
  python scripts/22_int8_quantize.py --model <ncnn_model_dir> --pixel RGB \
      --table results/int8/x.table --out-dir runs/.../best_ncnn_model_int8
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import ROOT, need

NCNN_BIN = ROOT / "tools" / "ncnn-20261020-windows-vs2022-shared" / "x64" / "bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="FP32 ncnn 模型目录")
    ap.add_argument("--pixel", default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--method", default="kl", choices=["kl", "aciq", "eq", "percentile"])
    ap.add_argument("--table", required=True, help="校准表输出路径")
    ap.add_argument("--out-dir", required=True, help="INT8 目录（须含 _ncnn_model）")
    ap.add_argument("--calib-list", default="data/int8_calib/images_list.txt")
    args = ap.parse_args()

    model_dir = Path(args.model)
    model_dir = model_dir if model_dir.is_absolute() else ROOT / model_dir
    param, bin_ = model_dir / "model.ncnn.param", model_dir / "model.ncnn.bin"
    need(param.exists() and bin_.exists(), f"ncnn param/bin 不存在: {model_dir}")

    table = Path(args.table)
    table = table if table.is_absolute() else ROOT / table
    table.parent.mkdir(parents=True, exist_ok=True)

    calib = Path(args.calib_list)
    calib = calib if calib.is_absolute() else ROOT / calib
    need(calib.exists(), f"校准图列表不存在: {calib}")

    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    need("_ncnn_model" in out_dir.name, "out-dir 名称须含 _ncnn_model（ultralytics NCNN 识别）")

    # 1) ncnn2table
    cmd = [str(NCNN_BIN / "ncnn2table.exe"), str(param), str(bin_), str(calib), str(table),
           "mean=[0,0,0]", "norm=[0.0039215686,0.0039215686,0.0039215686]",
           "shape=[640,640,3]", f"pixel={args.pixel}", "thread=4", f"method={args.method}", "type=0"]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, errors="replace")
    need(r.returncode == 0, f"ncnn2table 失败: {r.stdout[-400:]}{r.stderr[-400:]}")
    print(f"[table] {table} (pixel={args.pixel}, method={args.method})")

    # 2) ncnn2int8
    tmp_param = table.parent / f"model_int8_{args.pixel.lower()}.param"
    tmp_bin = table.parent / f"model_int8_{args.pixel.lower()}.bin"
    r = subprocess.run([str(NCNN_BIN / "ncnn2int8.exe"), str(param), str(bin_),
                        str(tmp_param), str(tmp_bin), str(table)],
                       cwd=str(ROOT), capture_output=True, text=True, errors="replace")
    need(r.returncode == 0, f"ncnn2int8 失败: {r.stdout[-300:]}")

    # 3) 组装
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(model_dir, out_dir)
    shutil.copy2(tmp_param, out_dir / "model.ncnn.param")
    shutil.copy2(tmp_bin, out_dir / "model.ncnn.bin")
    size_mb = round(sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6, 3)
    print(f"[int8] {out_dir}  size={size_mb} MB")


if __name__ == "__main__":
    main()
