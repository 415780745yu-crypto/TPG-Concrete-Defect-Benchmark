# -*- coding: utf-8 -*-
"""42: 源域 vs 目标域外观对比图（图 7）——展示域偏移的直观依据。

用途：论文当前缺少"两个域长什么样"的可视化，而 0.6501 的域间隙是全文前提。
本图取两域各 3 张典型图，标注缺陷密度与外观差异。

选图原则（可复现，非挑好看的）：
  源域 CUBIT-Det：按文件名序取前 3 张（hk0000–0002）
  目标域 DamCrack：取测试集前 3 张（A1, A10, A100），与失败案例图同源
缺陷密度由 label txt 统计，不靠人眼判断。

产物：
  results/figs/fig7_domain_appearance.pdf / .png
  results/tables/fig7_appearance.csv

用法：python scripts/42_domain_appearance.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figs"

SRC_IMG = Path("D:/YaLi/CUBIT-Det/images")
SRC_LBL = Path("D:/YaLi/CUBIT-Det/labels")
TGT_IMG = Path("D:/YaLi/Damage Detection/images")
TGT_LBL = Path("D:/YaLi/Damage Detection/labels/Yolo")

CELL = 380
HDR = 30
PAD = 14
CAP = 26
FOOT = 52


def get_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def count_boxes(lbl: Path) -> tuple[int, dict[int, int]]:
    """返回 (总框数, {类别: 框数})。"""
    if not lbl.exists():
        return 0, {}
    per: dict[int, int] = {}
    n = 0
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                cid = int(float(p[0]))
            except ValueError:
                continue
            per[cid] = per.get(cid, 0) + 1
            n += 1
    return n, per


def main() -> None:
    for d in (SRC_IMG, TGT_IMG):
        if not d.is_dir():
            raise SystemExit(f"缺少图像目录 {d}")

    picks = [
        ("Source: CUBIT-Det", SRC_IMG, SRC_LBL,
         sorted(SRC_IMG.glob("*.jpg"))[:3]),
        ("Target: DamCrack", TGT_IMG, TGT_LBL,
         sorted(TGT_IMG.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))[:3]),
    ]

    out_w = CELL * 3 + PAD * 4
    out_h = (CELL + HDR + CAP) * 2 + FOOT + PAD * 3
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_hdr = get_font(24)
    f_cap = get_font(17)
    f_foot = get_font(19)

    rows_out = []
    for r, (title, idir, ldir, files) in enumerate(picks):
        y = PAD + r * (CELL + HDR + CAP + PAD)
        # 行标题
        draw.rectangle([PAD, y, out_w - PAD, y + HDR - 4], fill=(235, 240, 248))
        draw.text((PAD + 8, y + 3), title, fill=(20, 40, 80), font=f_hdr)
        for c, f in enumerate(files):
            x = PAD + c * (CELL + PAD)
            im = Image.open(f).convert("RGB")
            im.thumbnail((CELL, CELL), Image.LANCZOS)
            # letterbox：不拉伸，居中放到 CELL×CELL 的白底上
            box = Image.new("RGB", (CELL, CELL), (255, 255, 255))
            box.paste(im, ((CELL - im.size[0]) // 2, (CELL - im.size[1]) // 2))
            canvas.paste(box, (x, y + HDR))
            n, per = count_boxes(ldir / f"{f.stem}.txt")
            cls = ", ".join(f"c{k}={v}" for k, v in sorted(per.items())) or "no boxes"
            cap = f"{f.name}  ·  {im.size[0]}×{im.size[1]}  ·  {n} boxes ({cls})"
            draw.text((x + 2, y + HDR + CELL + 2), cap, fill=(60, 60, 60), font=f_cap)
            rows_out.append({"domain": title.split(":")[0], "image": f.name,
                             "w": im.size[0], "h": im.size[1],
                             "n_boxes": n,
                             "per_class": ";".join(f"{k}:{v}" for k, v in sorted(per.items()))})

    foot1 = ("Note different annotation protocols: CUBIT-Det has 3 classes "
             "(c0 crack, c1 spalling, c2 moisture); DamCrack has 2 (c0 crack, c1 spalling).")
    foot2 = ("Images are letterboxed to a common cell size for display only, "
             "not rescaled in the experiments.")
    draw.text((PAD, out_h - FOOT + 2), foot1, fill=(70, 70, 70), font=f_foot)
    draw.text((PAD, out_h - FOOT + 24), foot2, fill=(70, 70, 70), font=f_foot)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig7_domain_appearance.png"
    canvas.save(png, optimize=True)
    print(f"[write] {png}  ({png.stat().st_size/1024:.0f} KB)")
    pdf = OUT / "fig7_domain_appearance.pdf"
    canvas.save(pdf, "PDF", resolution=250.0)
    print(f"[write] {pdf}")

    with open(RESULTS / "tables" / "fig7_appearance.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "image", "w", "h", "n_boxes", "per_class"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"[write] {RESULTS/'tables'/'fig7_appearance.csv'}")
    for r in rows_out:
        print(f"  {r['domain']:<10} {r['image']:<12} {r['n_boxes']:>3} boxes  [{r['per_class']}]")


if __name__ == "__main__":
    main()
