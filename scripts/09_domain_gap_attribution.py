"""方案 A：域间隙敏感性分析（主）+ 方案 B：标注预算派生指标（零成本后处理）。

目的
----
在 zero-shot 评测之后，**不改模型、不重训**，对 DamCrack 测试集施加四组受控变换，
各评一次 mAP50^T，把零样本掉点按「成像 / 尺度 / 纹理」做敏感性归因。
落在论文 §4.4 与 Fig 4 右栏；同时从已有恢复曲线派生标注预算结论，落在 Fig 5 与 Abstract。

四组变换（标签是归一化 YOLO txt，所有变换都保持原图尺寸，标签直接复制）
  T0_original     原始测试图
  T1_photometric  把目标域全局光度统计（逐通道 mean/std）仿射对齐到源域 CUBIT-Det
  T2_scale        下采样 ×scale_factor 再上采样回原尺寸，模拟尺度/分辨率差异
  T3_texture      灰度化 + 轻度高斯模糊，模拟纹理/清晰度差异

前提
----
  1. 先跑 01_prepare_data.py 生成 data/splits/damcrack_test.txt 与 cubit_det_train.txt
  2. 先跑 03_train.py + 04_eval.py --protocol crossdomain 得到 zero-shot 与 target_only 结果
  3. 环境已装 ultralytics（Anaconda base 的 torch 2.4.0+cu118 可直接装）

用法
----
  # 零成本先出 Fig 4 频域统计（不需要权重、不需要 GPU，训练前就能跑）
  python scripts/09_domain_gap_attribution.py --spectral-only

  # 先小样本验证链路（强烈建议第一次这样跑）
  python scripts/09_domain_gap_attribution.py --weights <源域训练的 best.pt> --limit 100

  # 正式跑全测试集
  python scripts/09_domain_gap_attribution.py --weights <源域训练的 best.pt>

  # 只做方案 B（不需要权重、不需要 GPU，纯后处理）
  python scripts/09_domain_gap_attribution.py --budget-only

输出（全部写入已有的 results/ 目录，不新建项目文档）
  results/domain_gap.csv              四组 mAP50 / 分类别 AP / 相对 T0 变化
  results/tables/fig4b_domain_gap.csv Fig 4 右栏柱状图数据
  results/fig4b_domain_gap.png        Fig 4 右栏柱状图
  results/tables/fig4_spectral.csv    两域平均径向频谱能量曲线（Fig 4 第三栏）
  results/fig4_spectral.png           两域频谱对比图
  results/annotation_budget.csv       方案 B：达到 90% 上界所需 shot 数、每类恢复斜率
  results/tables/fig5_90pct.csv       Fig 5 的 90% 参考线与各 shot 点

频域统计的性质
--------------
频域统计是**描述性分析**，不是模块、不进网络、不参与推理、不影响导出链。
用 numpy FFT 计算两域平均径向频谱能量曲线，为 §3.2 / Fig 4 补充一个域间隙描述维度，
并呼应 T3 纹理扰动。禁止写成「频域是掉点原因」「频域增强提升精度」之类的因果断言。

措辞纪律（写论文时必须遵守）
----------------------------
这是 **sensitivity analysis（敏感性分析）**，不是新的域自适应算法。
只能写「该变换部分缓解掉点 / 掉点对某因素敏感」，**禁止**写「消除域间隙」「证明掉点来自 X」
或把各变换的恢复量相加（变换之间不独立，不构成因果分解）。
所有数字必须来自本脚本实测，禁止估算。
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from common import ROOT, RESULTS, abs_path, load_cfg, need

VARIANTS = ("T0_original", "T1_photometric", "T2_scale", "T3_texture")


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise SystemExit("[STOP] 缺少 opencv-python，请先执行 pip install -r requirements.txt")


def _imread(cv2, p: Path):
    """Windows 中文路径下 cv2.imread 会失败，统一走 imdecode。"""
    buf = np.fromfile(str(p), dtype=np.uint8)
    need(buf.size > 0, f"无法读取图像：{p}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    need(img is not None, f"图像解码失败：{p}")
    return img


def _imwrite(cv2, p: Path, img, quality: int):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    need(ok, f"图像编码失败：{p}")
    buf.tofile(str(p))


def global_stats(cv2, paths, sample: int, seed: int = 42):
    """抽样计算逐通道全局 mean/std，用于固定仿射光度对齐（保持图像间相对差异）。"""
    import random

    pool = list(paths)
    if len(pool) > sample:
        pool = random.Random(seed).sample(pool, sample)
    total = np.zeros(3, np.float64)
    sq = np.zeros(3, np.float64)
    n = 0
    for p in pool:
        px = _imread(cv2, p).astype(np.float64).reshape(-1, 3)
        total += px.sum(0)
        sq += (px ** 2).sum(0)
        n += px.shape[0]
    need(n > 0, "全局统计抽样为空")
    mean = total / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-6))
    return mean, std


def t_photometric(cv2, bgr, src_mean, src_std, tgt_mean, tgt_std):
    out = ((bgr.astype(np.float32) - tgt_mean.reshape(1, 1, 3))
           / tgt_std.reshape(1, 1, 3)
           * src_std.reshape(1, 1, 3) + src_mean.reshape(1, 1, 3))
    return np.clip(out, 0, 255).astype(np.uint8)


def t_scale(cv2, bgr, factor: int):
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def t_texture(cv2, bgr, kernel: int):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)


def build_variant(cv2, tag, images, label_dir, out_root, names, fn, quality, force):
    vdir = out_root / tag
    idir, ldir, yaml = vdir / "images", vdir / "labels", vdir / "data.yaml"
    if yaml.exists() and not force:
        print(f"[reuse] {tag}（已存在，加 --force 可重建）")
        return yaml
    if vdir.exists():
        shutil.rmtree(vdir)
    idir.mkdir(parents=True)
    ldir.mkdir(parents=True)
    for p in images:
        bgr = _imread(cv2, p)
        _imwrite(cv2, idir / p.name, fn(bgr) if fn else bgr, quality)
        lb = label_dir / f"{p.stem}.txt"
        need(lb.exists(), f"缺少标签 {lb}；请确认 configs/datasets.yaml 的 label_dir 指向 labels/Yolo")
        shutil.copy2(lb, ldir / lb.name)
    lines = [f"path: {vdir.as_posix()}", "train: images", "val: images",
             "test: images", f"nc: {len(names)}", "names:"]
    lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[build] {tag}: {len(images)} 张")
    return yaml


def evaluate(model, yaml, cfg, tag, n_images, val_logs):
    m = model.val(data=str(yaml), split="test", imgsz=cfg["imgsz"],
                  verbose=False, project=str(val_logs), name=tag, exist_ok=True)
    box = m.box
    row = {"variant": tag, "n_images": n_images,
           "map50": round(float(box.map50), 4), "map5095": round(float(box.map), 4)}
    try:
        for i, name in m.names.items():
            row[f"ap50_{name}"] = round(float(box.ap50[i]), 4)
    except Exception as exc:
        print(f"[warn] {tag} 分类别 AP 未取得：{exc}")
    return row


def write_csv(rows, path: Path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path}")


def plot_fig4b(rows, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] 缺少 matplotlib，跳过柱状图；CSV 数据仍已生成")
        return
    tags = [r["variant"] for r in rows]
    vals = [r["map50"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(tags, vals, color="#4C78A8", width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("mAP50 on DamCrack test")
    ax.set_title("Zero-shot sensitivity to controlled perturbations")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[write] {out}")


def run_attribution(args, cfg):
    need(args.weights, "方案 A 需要 --weights 指定源域训练得到的权重（zero-shot 用的那个 best.pt）")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 缺少 ultralytics。请在 Anaconda base 执行："
                         "C:/Users/zhenbai/anaconda3/python.exe -m pip install ultralytics")

    cv2 = _cv2()
    dg = cfg.get("domain_gap", {})
    sample = int(dg.get("stat_sample", 200))
    factor = int(dg.get("scale_factor", 2))
    kernel = int(dg.get("blur_kernel", 3))
    quality = int(dg.get("jpeg_quality", 95))

    dc = cfg["datasets"]["damcrack"]
    root = abs_path(dc["root"])
    label_dir = root / dc["label_dir"]
    test_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    need(test_txt.exists(), "请先执行 01_prepare_data.py 生成 data/splits/damcrack_test.txt")
    images = [Path(x) for x in test_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    need(len(images) > 0, "测试集列表为空")
    need(label_dir.is_dir(), f"标签目录不存在：{label_dir}")

    # 源域全局光度统计：缺失则跳过 T1，绝不伪造对齐
    src_txt = ROOT / "data" / "splits" / "cubit_det_train.txt"
    src_mean = src_std = None
    if src_txt.exists():
        src_imgs = [Path(x) for x in src_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
        if src_imgs:
            src_mean, src_std = global_stats(cv2, src_imgs, sample)
            print(f"[stats] source mean={src_mean.round(1)} std={src_std.round(1)}")
        else:
            print("[warn] 源域划分文件为空，T1 跳过")
    else:
        print("[warn] 未找到 data/splits/cubit_det_train.txt，T1 跳过（不伪造光度对齐）")
    tgt_mean, tgt_std = global_stats(cv2, images, sample)
    print(f"[stats] target mean={tgt_mean.round(1)} std={tgt_std.round(1)}")

    variants = {
        "T0_original": None,
        "T2_scale": lambda b: t_scale(cv2, b, factor),
        "T3_texture": lambda b: t_texture(cv2, b, kernel),
    }
    if src_mean is not None:
        variants["T1_photometric"] = lambda b: t_photometric(
            cv2, b, src_mean, src_std, tgt_mean, tgt_std)

    out_root = RESULTS / "domain_gap" / "variants"
    val_logs = RESULTS / "domain_gap" / "val_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    rows = []
    for tag in VARIANTS:
        if tag not in variants:
            print(f"[skip] {tag}（前置条件未满足）")
            continue
        yaml = build_variant(cv2, tag, images, label_dir, out_root,
                             cfg["classes"]["target"], variants[tag], quality, args.force)
        row = evaluate(model, yaml, cfg, tag, len(images), val_logs)
        rows.append(row)
        print(f"[eval] {tag}: mAP50={row['map50']}")

    need(rows, "没有任何变体成功评估")
    base = next((r["map50"] for r in rows if r["variant"] == "T0_original"), None)
    upper = None
    cross = RESULTS / "crossdomain.csv"
    if cross.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cross)
            hit = df[df["tag"].astype(str).str.contains("target_only", case=False, na=False)]
            if len(hit):
                upper = float(hit["map50"].iloc[-1])
        except Exception as exc:
            print(f"[warn] 读取 target_only 上界失败：{exc}")

    for r in rows:
        r["delta_vs_T0"] = round(r["map50"] - base, 4) if base is not None else ""
        if upper is not None and base is not None and upper > base:
            r["recovery_ratio"] = round((r["map50"] - base) / (upper - base), 4)
            r["recovery_note"] = "fraction of gap to target-only upper bound"
        else:
            r["recovery_ratio"] = ""
            r["recovery_note"] = "upper bound unavailable"
        r["caveat"] = "sensitivity only; variants are not independent, do not sum"

    write_csv(rows, RESULTS / "domain_gap.csv")
    write_csv(rows, RESULTS / "tables" / "fig4b_domain_gap.csv")
    plot_fig4b(rows, RESULTS / "fig4b_domain_gap.png")
    print("[提醒] 论文措辞：sensitivity analysis / partially alleviates；禁止写「消除域间隙」或把各变换恢复量相加")


def run_budget(cfg):
    """方案 B：从已有恢复曲线派生标注预算结论，零训练、零 GPU。"""
    cross = RESULTS / "crossdomain.csv"
    if not cross.exists():
        print("[缺少] results/crossdomain.csv —— 请先跑 03_train.py + 04_eval.py --protocol crossdomain")
        return
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("[STOP] 缺少 pandas，请先 pip install -r requirements.txt")
    df = pd.read_csv(cross)
    need("map50" in df.columns, "crossdomain.csv 缺少 map50 列")

    def pick(keyword):
        hit = df[df["tag"].astype(str).str.contains(keyword, case=False, na=False)]
        return float(hit["map50"].iloc[-1]) if len(hit) else None

    lower, upper = pick("zero"), pick("target_only")
    grouped = {}
    for _, r in df.iterrows():
        m = re.search(r"shot(\d+)", str(r["tag"]))
        if m:
            grouped.setdefault(int(m.group(1)), []).append(float(r["map50"]))
    points = [(s, float(np.mean(v))) for s, v in sorted(grouped.items())]
    need(bool(points), "crossdomain.csv 中没有 shot 记录（tag 需含 shot10 / shot20 等）")

    rows = []
    for s, m in points:
        ratio = ((m - lower) / (upper - lower)) if (lower is not None and upper and upper > lower) else None
        rows.append({"shots": s, "map50": round(m, 4),
                     "recovery_ratio": round(ratio, 4) if ratio is not None else "",
                     "recoverable_gap": round(upper - lower, 4) if (lower is not None and upper) else "",
                     "note": "mean over 3 seeds; no extrapolation beyond tested range"})
    write_csv(rows, RESULTS / "annotation_budget.csv")

    shot50_mean = next((m for s, m in points if s == 50), None)
    shot50_ratio = None
    if shot50_mean is not None and upper:
        shot50_ratio = shot50_mean / upper if upper > 0 else None
    summary = {"target_only_upper": round(upper, 4) if upper else "",
               "zero_shot_lower": round(lower, 4) if lower else "",
               "shot50_map50": round(shot50_mean, 4) if shot50_mean is not None else "",
               "shot50_recovery_ratio_pct": round(shot50_ratio * 100, 1) if shot50_ratio is not None else "",
               "note_90pct": (
                   f"50-shot 仅恢复到上界 {round(upper,4) if upper else 'N/A'} 的 "
                   f"{round(shot50_ratio*100,1) if shot50_ratio is not None else 'N/A'}%；"
                   "达到 90% 所需标注量远超本文考察范围，不做线性外推"
               )}
    write_csv([summary], RESULTS / "tables" / "fig5_90pct.csv")

    # 每类 AP 恢复斜率（若 04_eval.py 已写入分类别列）
    slopes = []
    for cls in ("crack", "spalling"):
        col = f"ap50_{cls}"
        if col not in df.columns:
            continue
        vals = []
        for s, _ in points:
            sub = df[df["tag"].astype(str).str.contains(f"shot{s}", na=False)]
            if len(sub):
                vals.append(float(sub[col].mean()))
        if len(vals) >= 2:
            slopes.append({"class": cls, "ap_slope_per_shot": round(float(np.polyfit(
                [s for s, _ in points[:len(vals)]], vals, 1)[0]), 6)})
    if slopes:
        write_csv(slopes, RESULTS / "tables" / "class_recovery_slope.csv")

    print(f"[budget] 50-shot 恢复到上界的 {summary['shot50_recovery_ratio_pct']}%；"
          "达到 90% 所需标注量远超本文考察范围（不做线性外推）")


def run_source_control(args, cfg):
    """源域对照：把 T2_scale / T3_texture 以目标域侧完全相同的参数施加到 CUBIT-Det 测试集
    （源域 3 类，基线 0.7153），各评一次。结果写独立文件 results/source_control.csv，
    不与目标域 2 类的 domain_gap.csv 混表。"""
    need(args.weights, "--source-control 需要 --weights 指定源域权重")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 缺少 ultralytics。请在 Anaconda base 执行 pip install -r requirements.txt")

    cv2 = _cv2()
    dg = cfg.get("domain_gap", {})
    factor = int(dg.get("scale_factor", 2))
    kernel = int(dg.get("blur_kernel", 3))
    quality = int(dg.get("jpeg_quality", 95))

    sc = cfg["datasets"]["cubit_det"]
    root = abs_path(sc["root"])
    label_dir = root / sc["yolo_label_dir"]
    test_txt = ROOT / "data" / "splits" / "cubit_det_test.txt"
    need(test_txt.exists(), "缺少 data/splits/cubit_det_test.txt，先跑 01_prepare_data.py")
    images = [Path(x) for x in test_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    need(len(images) > 0, "源域测试集列表为空")
    names = cfg["classes"]["source"]

    out_root = RESULTS / "source_control" / "variants"
    val_logs = RESULTS / "source_control" / "val_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    variants = {
        "T2_scale": lambda b: t_scale(cv2, b, factor),
        "T3_texture": lambda b: t_texture(cv2, b, kernel),
    }
    rows = []
    for tag, fn in variants.items():
        yaml = build_variant(cv2, tag, images, label_dir, out_root, names, fn, quality, args.force)
        row = evaluate(model, yaml, cfg, tag, len(images), val_logs)
        rows.append(row)

    baseline = None
    cross = RESULTS / "crossdomain.csv"
    if cross.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cross)
            hit = df[df["tag"].astype(str).str.contains("source_only", case=False, na=False)]
            if len(hit):
                baseline = float(hit["map50"].iloc[-1])
        except Exception as exc:
            print(f"[warn] 读取 source_only 基线失败：{exc}")
    for r in rows:
        r["baseline_map50"] = round(baseline, 4) if baseline else ""
        r["delta_vs_baseline"] = round(r["map50"] - baseline, 4) if baseline is not None else ""
        r["note"] = "source-control: 与目标域侧参数一致，施加到 CUBIT-Det 源域测试集（3 类）"
    write_csv(rows, RESULTS / "source_control.csv")
    print("[source-control] 完成 -> results/source_control.csv")
    for r in rows:
        print(f"  {r['variant']}: mAP50={r['map50']}  baseline={r['baseline_map50']}  delta={r['delta_vs_baseline']}")


def _radial_profile(mag, bins=64):
    """把二维幅度谱按半径做圆周平均，得到归一化能量分布曲线。"""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32).ravel()
    cnt = np.bincount(r, minlength=bins)
    s = np.bincount(r, weights=mag.ravel(), minlength=bins)
    prof = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)[:bins]
    return prof / (prof.sum() + 1e-9)


def _mean_spectrum(cv2, paths, sample, seed=42, size=256, bins=64):
    import random

    pool = list(paths)
    if len(pool) > sample:
        pool = random.Random(seed).sample(pool, sample)
    acc = np.zeros(bins, np.float64)
    ok = 0
    for p in pool:
        bgr = _imread(cv2, p)
        gray = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2GRAY)
        g = gray.astype(np.float32) / 255.0
        mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
        acc += _radial_profile(mag, bins)
        ok += 1
    need(ok > 0, "频谱统计抽样为空")
    return acc / ok


def run_spectral(cfg):
    """Fig 4 第三栏：两域平均径向频谱能量曲线。纯描述性分析，零训练、不进网络。"""
    cv2 = _cv2()
    sample = int(cfg.get("domain_gap", {}).get("stat_sample", 200))
    src_txt = ROOT / "data" / "splits" / "cubit_det_train.txt"
    tgt_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    if not src_txt.exists() or not tgt_txt.exists():
        print("[skip] 频谱统计需要 01_prepare_data.py 生成的两个 split 文件")
        return
    src = [Path(x) for x in src_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    tgt = [Path(x) for x in tgt_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not src or not tgt:
        print("[skip] split 文件为空")
        return
    print("[spectral] 计算两域平均径向频谱（纯分析，不进网络）...")
    s = _mean_spectrum(cv2, src, sample)
    t = _mean_spectrum(cv2, tgt, sample)
    rows = [{"radius_bin": i,
             "source_CUBIT_Det": round(float(s[i]), 8),
             "target_DamCrack": round(float(t[i]), 8),
             "log2_ratio_target_over_source": round(float(np.log2((t[i] + 1e-9) / (s[i] + 1e-9))), 4),
             "note": "descriptive only; no causal claim"}
            for i in range(len(s))]
    write_csv(rows, RESULTS / "tables" / "fig4_spectral.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        x = np.arange(len(s))
        ax.plot(x, s, label="CUBIT-Det (source)", color="#4C78A8")
        ax.plot(x, t, label="DamCrack (target)", color="#F58518")
        ax.set_yscale("log")
        ax.set_xlabel("radial frequency bin")
        ax.set_ylabel("normalized spectral energy (log)")
        ax.set_title("Mean radial spectral energy: descriptive domain-gap view")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig4_spectral.png", dpi=150)
        plt.close(fig)
        print(f"[write] {RESULTS / 'fig4_spectral.png'}")
    except Exception as exc:
        print(f"[warn] 频谱图未保存: {exc}")
    print("[提醒] 论文措辞只写「观察到频谱分布差异 / 与该因素相容」，禁止因果断言或包装成新方法")


def main():
    ap = argparse.ArgumentParser(description="域间隙敏感性分析 + 频域统计 + 标注预算派生指标")
    ap.add_argument("--weights", default=None, help="源域训练得到的 best.pt（zero-shot 权重）")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 张测试图，0=全部；首次建议 100")
    ap.add_argument("--force", action="store_true", help="强制重建变体图像目录")
    ap.add_argument("--budget-only", action="store_true", help="只跑方案 B，不需要权重与 GPU")
    ap.add_argument("--spectral-only", action="store_true", help="只跑频域统计，不需要权重与 GPU")
    ap.add_argument("--source-control", action="store_true", help="把 T2/T3 施加到 CUBIT-Det 源域测试集做对照")
    args = ap.parse_args()
    cfg = load_cfg()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.spectral_only:
        run_spectral(cfg)
        return
    if args.budget_only:
        run_budget(cfg)
        return
    if args.source_control:
        run_source_control(args, cfg)
        return
    run_attribution(args, cfg)
    print("[auto] 顺带执行频域统计与方案 B（零成本后处理）")
    run_spectral(cfg)
    run_budget(cfg)


if __name__ == "__main__":
    main()
