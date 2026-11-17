# -*- coding: utf-8 -*-
"""00_check_env.py — 只读自检：解释器 / torch / CUDA / ultralytics / 数据 / 划分 / 已有产出。

用法（必须用 Anaconda base 的 python，不要用 pytorch env）：
    C:/Users/zhenbai/anaconda3/python.exe scripts/00_check_env.py

本脚本不训练、不写任何文件，只打印状态。退出码 0 = 可开训，2 = 有阻塞项。
"""
from __future__ import annotations

import glob
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BLOCKERS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    BLOCKERS.append(msg)


def section(title: str) -> None:
    print(f"\n[{title}]")


def count_images(d: Path) -> int:
    if not d.exists():
        return -1
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    print("=" * 72)
    print("PW-YOLO 环境与产出物自检（只读）")
    print("=" * 72)

    section("0 项目地图")
    print(f"  项目根目录 : {ROOT}")
    for rel in ("scripts", "configs", "data", "results"):
        p = ROOT / rel
        print(f"  {rel:11s}: {'存在' if p.exists() else '缺失'}  {p}")

    # ---------- 1 解释器 ----------
    section("1 Python 解释器")
    exe = sys.executable
    print(f"  executable : {exe}")
    print(f"  version    : {sys.version.split()[0]}  ({platform.architecture()[0]})")
    low = exe.lower().replace("/", "\\")
    if "\\envs\\pytorch" in low:
        bad("当前是 pytorch 环境（torch 1.13.1+cu116），本项目禁用；请改用 C:/Users/zhenbai/anaconda3/python.exe")
    elif "anaconda3\\python.exe" in low:
        ok("使用的是 Anaconda base（定稿基线）")
    else:
        warn("不在 anaconda3 base 里；定稿基线是 C:/Users/zhenbai/anaconda3/python.exe")

    # ---------- 2 torch / CUDA ----------
    section("2 PyTorch / CUDA")
    try:
        import torch  # noqa: PLC0415

        print(f"  torch      : {torch.__version__}")
        print(f"  cuda_build : {torch.version.cuda}")
        avail = torch.cuda.is_available()
        print(f"  available  : {avail}")
        if avail:
            print(f"  device     : {torch.cuda.get_device_name(0)}")
            try:
                print(f"  cudnn      : {torch.backends.cudnn.version()}")
            except Exception:
                pass
            if torch.version.cuda and not str(torch.version.cuda).startswith("11"):
                warn("cuda_build 不是 11.x：可能被动升级过，见备忘第 12 节")
            else:
                ok("GPU 可用，CUDA 构建符合 11.x 基线")
        else:
            bad("GPU 不可用：禁止开训，先排查 torch 是否被换成 CPU 版（备忘第 12 节）")
    except Exception as e:  # pragma: no cover
        bad(f"import torch 失败：{e}")

    # ---------- 3 ultralytics / onnx ----------
    section("3 ultralytics / onnx")
    try:
        import ultralytics  # noqa: PLC0415

        print(f"  ultralytics: {ultralytics.__version__}")
        if ultralytics.__version__ == "8.2.103":
            ok("与定稿基线一致（8.2.103）")
        else:
            warn("与定稿基线 8.2.103 不同：不要无约束升级（会触发 torch 被替换，见备忘第 12 节）")
    except Exception as e:
        bad(f"import ultralytics 失败：{e}")

    try:
        import onnx  # noqa: F401, PLC0415

        warn(f"检测到 onnx {getattr(onnx, '__version__', '?')}：base 里不该有它（与 streamlit 的 protobuf 冲突）；导出链应在独立环境做")
    except ImportError:
        ok("onnx 未安装（符合定稿：留到 Step 8 用独立环境）")

    # ---------- 4 数据 ----------
    section("4 数据与配置")
    try:
        from common import load_cfg  # noqa: PLC0415

        cfg = load_cfg()
        for key in ("cubit_det", "damcrack"):
            d = cfg["datasets"][key]
            root = Path(d["root"])
            img_dir = root / d["image_dir"]
            lab_dir = root / d.get("yolo_label_dir", d["label_dir"])
            n_img = count_images(img_dir)
            n_lab = len(list(lab_dir.glob("*.txt"))) if lab_dir.exists() else -1
            print(f"  {key:10s} root   = {root}")
            print(f"  {'':10s} images = {n_img}   ({img_dir.name}/)")
            print(f"  {'':10s} labels = {n_lab}   ({lab_dir.relative_to(root) if lab_dir.is_relative_to(root) else lab_dir})")
            if n_img <= 0 or n_lab <= 0:
                bad(f"{key}: 图像或标签目录为空，检查 root / image_dir / label_dir")
            elif n_img != n_lab:
                warn(f"{key}: 图像({n_img}) 与标签({n_lab}) 数量差 {abs(n_img - n_lab)}（空标签/负样本属正常，确认后可忽略）")
            else:
                ok(f"{key}: 图像与标签一一对应（{n_img}）")
        w = ROOT / cfg.get("baseline_weights", "yolov8n.pt")
        ok(f"预训练权重存在：{w.name}（{w.stat().st_size / 1e6:.1f} MB）") if w.exists() else warn(f"缺预训练权重 {w}")
    except Exception as e:
        bad(f"读取 configs/datasets.yaml 失败：{e}")

    # ---------- 5 划分 ----------
    section("5 划分文件 data/splits")
    expect = {
        "cubit_det_train": 3867, "cubit_det_val": 828, "cubit_det_test": 830,
        "damcrack_train": 2800, "damcrack_val": 600, "damcrack_test": 600,
    }
    for name, exp in expect.items():
        p = ROOT / "data" / "splits" / f"{name}.txt"
        if not p.exists():
            bad(f"缺 {p.name}")
            continue
        n = sum(1 for _ in p.open(encoding="utf-8"))
        flag = "OK" if n == exp else "WARN"
        print(f"  [{flag}] {name:18s} {n:5d} 行  (基线 {exp})")
        if flag == "WARN":
            warn(f"{name} 行数与基线不符，确认是否重建过划分")
    for y in ("cubit_det.yaml", "damcrack.yaml"):
        (ok if (ROOT / "data" / y).exists() else bad)(f"data/{y}")

    # ---------- 6 已有产出 ----------
    section("6 已有产出物")
    artifacts = [
        "results/split_protocol.md", "results/class_protocol.md",
        "results/split_summary.csv", "results/tables/fig4_spectral.csv",
        "results/fig4_spectral.png",
    ]
    for rel in artifacts:
        p = ROOT / rel
        (ok if p.exists() else warn)(rel)

    ckpts = sorted(glob.glob(str(ROOT / "runs" / "*" / "weights" / "best.pt")))
    if ckpts:
        ok(f"runs/ 中已有 {len(ckpts)} 个 best.pt：")
        for c in ckpts:
            print(f"         {Path(c).relative_to(ROOT)}")
    else:
        warn("runs/ 为空：正式训练（Step 3）尚未运行")

    # ---------- 7 下一步 ----------
    section("7 下一步")
    if ckpts:
        print("  已有 source_pretrain 权重，继续跑跨域评估：")
        print("  C:/Users/zhenbai/anaconda3/python.exe scripts/04_eval.py --weights runs/pw_yolo_source_pretrain/weights/best.pt --protocol crossdomain")
    else:
        print("  C:/Users/zhenbai/anaconda3/python.exe scripts/03_train.py --data data/cubit_det.yaml --tag source_pretrain")

    print("\n" + "=" * 72)
    if BLOCKERS:
        print(f"结论：{len(BLOCKERS)} 项阻塞，先修再开训")
        for b in BLOCKERS:
            print(f"  - {b}")
        return 2
    print("结论：可开训（阻塞项 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
