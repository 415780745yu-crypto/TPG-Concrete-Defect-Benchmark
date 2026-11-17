"""19: 导出一致性校验（PyTorch / ONNX / NCNN 同一张真实图）。

对固定真实图用三种后端各推理一次，比较解码框的坐标与类别分数，
记录最大绝对差与判定到 results/consistency_check.csv，并追加一行到 export_log.csv。

用法：
  python scripts/19_consistency_check.py --pt <best.pt> --onnx <best.onnx> --ncnn <best_ncnn_model> \
      --source <真实图路径> --tag <model_label>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, RESULTS, append_csv, need

COORD_THR = 2.0     # 像素
SCORE_THR = 0.02    # 类别分数绝对差


def boxes_of(model, src, imgsz):
    res = model.predict(src, imgsz=imgsz, conf=0.001, verbose=False)[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy().tolist()
    conf = b.conf.cpu().numpy().tolist()
    cls = b.cls.cpu().numpy().astype(int).tolist()
    return sorted(zip(cls, conf, xyxy), key=lambda x: (x[0], -x[1]))


def compare(a, b):
    """a=PT, b=back；按 (cls) 分组、中心最近贪心配对，返回 (max_coord_diff, max_score_diff, n_a, n_b)。"""
    if not a and not b:
        return 0.0, 0.0, 0, 0
    max_coord = 0.0
    max_score = 0.0
    used = [False] * len(b)
    for (c1, s1, x1) in a:
        best_j, best_d = -1, 1e18
        for j, (c2, s2, x2) in enumerate(b):
            if used[j] or c1 != c2:
                continue
            cx1, cy1 = (x1[0] + x1[2]) / 2, (x1[1] + x1[3]) / 2
            cx2, cy2 = (x2[0] + x2[2]) / 2, (x2[1] + x2[3]) / 2
            d = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            c2, s2, x2 = b[best_j]
            max_coord = max(max_coord, max(abs(x1[k] - x2[k]) for k in range(4)))
            max_score = max(max_score, abs(s1 - s2))
    return round(max_coord, 3), round(max_score, 4), len(a), len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ncnn", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    src = args.source
    m_pt = YOLO(str(Path(args.pt)))
    m_on = YOLO(str(Path(args.onnx)))
    m_nc = YOLO(str(Path(args.ncnn)), task="detect")

    b_pt = boxes_of(m_pt, src, 640)
    b_on = boxes_of(m_on, src, 640)
    b_nc = boxes_of(m_nc, src, 640)

    c_on, s_on, n_pt, n_on = compare(b_pt, b_on)
    c_nc, s_nc, _, n_nc = compare(b_pt, b_nc)

    def verdict(c, s, dn):
        if dn == 0 and c <= COORD_THR and s <= SCORE_THR:
            return "PASS"
        return "CHECK" if c <= 4 and s <= 0.05 else "FAIL"

    v_on = verdict(c_on, s_on, abs(n_pt - n_on))
    v_nc = verdict(c_nc, s_nc, abs(n_pt - n_nc))

    rows = [
        {"tag": args.tag, "backend": "onnx", "n_boxes_pt": n_pt, "n_boxes_backend": n_on,
         "max_abs_coord_diff_px": c_on, "max_abs_score_diff": s_on, "verdict": v_on},
        {"tag": args.tag, "backend": "ncnn", "n_boxes_pt": n_pt, "n_boxes_backend": n_nc,
         "max_abs_coord_diff_px": c_nc, "max_abs_score_diff": s_nc, "verdict": v_nc},
    ]
    p = RESULTS / "consistency_check.csv"
    write_header = not p.exists()
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)

    summary = (f"ONNX: coord={c_on}px score={s_on} n={n_pt}/{n_on} -> {v_on}; "
               f"NCNN: coord={c_nc}px score={s_nc} n={n_pt}/{n_nc} -> {v_nc}")
    append_csv("export_log.csv", {
        "tag": args.tag, "model_label": args.tag, "weights": str(Path(args.pt)),
        "format": "consistency_check", "imgsz": 640, "dynamic": False,
        "status": ("PASS" if (v_on == "PASS" and v_nc == "PASS") else "CHECK/FAIL"),
        "artifact": str(args.source), "log": summary,
    })
    print(f"[ok] {args.tag}: {summary}")
    print(f"[write] results/consistency_check.csv + export_log.csv")


if __name__ == "__main__":
    main()
