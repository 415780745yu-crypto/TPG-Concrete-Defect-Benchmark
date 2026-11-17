# -*- coding: utf-8 -*-
"""41: 重制失败案例对比图（图 5b）——出版级。

来源：results/fig7_failure_cases.png（3840×2560）
      + results/failure_case_stats.csv（逐图 TP/FP/FN）
      + data/splits/damcrack_test.txt（图像名顺序，取前 12 张）

原图结构（已实测确认）：
  单张对比图 = 1280×640 = 左 640（zero-shot） + 右 640（shot50 微调）
  网格 = 3 列 × 4 行 = 12 张对比图
  拼接顺序遵循 imgs 列表（i//3 行, i%3 列）

产物：
  results/figs/fig5b_failure_cases.pdf / .png  （< 2 MB）
  results/tables/fig5b_panels.csv

用法：python scripts/41_remake_failure_cases.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figs"

SRC = RESULTS / "fig7_failure_cases.png"
STATS = RESULTS / "failure_case_stats.csv"
TEST_TXT = ROOT / "data" / "splits" / "damcrack_test.txt"

CELL = 640          # 单格边长
PAIR_W = CELL * 2   # 一张对比图宽度
N_COLS = 3
N_ROWS = 4
N_IMG = 12          # 取前 12 张（与原脚本 --n 12 一致）


def load_stats() -> dict[tuple[str, str], dict]:
    with open(STATS, "r", newline="", encoding="utf-8-sig") as f:
        return {(r["image"], r["model"]): r for r in csv.DictReader(f)}


def get_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    for p in (SRC, STATS, TEST_TXT):
        if not p.exists():
            raise SystemExit(f"缺少 {p}")

    stats = load_stats()
    imgs = [Path(p.strip()).name for p in
            TEST_TXT.read_text(encoding="utf-8").splitlines() if p.strip()][:N_IMG]
    print(f"图像顺序（前 {N_IMG}）：{imgs[:3]} ...")

    im = Image.open(SRC).convert("RGB")
    W, H = im.size
    assert W == N_COLS * PAIR_W and H == N_ROWS * CELL, \
        f"原图 {W}x{H} 与预期 {N_COLS*PAIR_W}x{N_ROWS*CELL} 不符"

    # ---- 布局：4 组放在 2 列 × 2 行（只选差异最明显的 4 组），留足标签空间 ----
    # 选取标准：shot50 相对 zero_shot 的 FN 减少量最大的前 4 张
    score = []
    for i, name in enumerate(imgs):
        z = stats.get((name, "zero_shot"), {})
        s = stats.get((name, "shot50"), {})
        try:
            gain = int(z["fn"]) - int(s["fn"])
        except (KeyError, ValueError):
            gain = 0
        score.append((gain, i, name))
    score.sort(reverse=True)
    chosen = score[:4]
    print(f"按 FN 减少量选出的 4 组：{[c[2] for c in chosen]}")

    pad_x, hdr, foot = 12, 34, 30
    out_w = PAIR_W * 2 + pad_x * 3
    out_h = (CELL + hdr) * 2 + foot + pad_x * 2
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_hdr = get_font(26)
    f_foot = get_font(21)

    rows_out = []
    for k, (gain, idx, name) in enumerate(chosen):
        r, c = divmod(k, 2)
        src_row, src_col = divmod(idx, N_COLS)
        x_src = src_col * PAIR_W
        y_src = src_row * CELL
        x_dst = pad_x + c * (PAIR_W + pad_x)
        y_dst = pad_x + r * (CELL + hdr + pad_x)

        for j, model in enumerate(("zero_shot", "shot50")):
            patch = im.crop((x_src + j * CELL, y_src,
                             x_src + (j + 1) * CELL, y_src + CELL))
            px = x_dst + j * CELL
            canvas.paste(patch, (px, y_dst + hdr))

            st = stats.get((name, model), {})
            tag = ("zero-shot" if model == "zero_shot" else "fine-tuned (shot50)")
            head = f"{name}  ·  {tag}   TP {st.get('tp','?')}  FP {st.get('fp','?')}  FN {st.get('fn','?')}"
            draw.rectangle([px, y_dst, px + CELL - 1, y_dst + hdr - 4], fill=(30, 30, 30))
            draw.text((px + 6, y_dst + 3), head, fill=(255, 255, 255), font=f_hdr)
            rows_out.append({"group": k, "image": name, "model": model,
                             "tp": st.get("tp"), "fp": st.get("fp"), "fn": st.get("fn")})

    legend = ("Green = ground truth    Red = prediction    "
              "Within each pair: left = zero-shot (source only), right = fine-tuned (shot50)")
    draw.text((pad_x, out_h - foot + 4), legend, fill=(50, 50, 50), font=f_foot)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig5b_failure_cases.png"
    canvas.convert("P", palette=Image.ADAPTIVE, colors=256).save(png, optimize=True)
    print(f"[write] {png}  ({png.stat().st_size/1024/1024:.2f} MB)")

    pdf = OUT / "fig5b_failure_cases.pdf"
    canvas.save(pdf, "PDF", resolution=200.0)
    print(f"[write] {pdf}  ({pdf.stat().st_size/1024/1024:.2f} MB)")

    with open(RESULTS / "tables" / "fig5b_panels.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "image", "model", "tp", "fp", "fn"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"[write] {RESULTS/'tables'/'fig5b_panels.csv'} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
