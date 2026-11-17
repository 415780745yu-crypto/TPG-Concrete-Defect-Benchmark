# -*- coding: utf-8 -*-
"""28: 用各档 A 组 best.pt 对无标注池打伪标签。

方案 A（伪标签自训练）第 1 步。τ 锁死 0.25，不做可配置（Pilot 纪律）。

复用 common.predict_to_txt()：该函数把 conf 直接传给
model.predict(conf=..., save_txt=True, save_conf=True)，
Ultralytics 在推理阶段即过滤低置信框，输出列为 cid cx cy w h conf。

输入：
  - 各档 A 组权重 runs/pw_yolo_<tier>_seed42/weights/best.pt
  - 无标注池 = data/splits/damcrack_train.txt (2800) 减去该档标注子集
输出：
  - results/predictions/pseudo_<tier>_seed42/labels/labels/*.txt

运行（CPU 慢，建议 GPU）：
  python scripts/28_pseudo_label.py                 # 全 7 档
  python scripts/28_pseudo_label.py --tiers shot50 # 单档调试
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, RESULTS, predict_to_txt

# τ 锁死：0.25 是 st500_thr025 / thr010 实验锚定的工程折中点。
# 0.1 引入大量假阳性噪声（0.0815），0.5 漏掉过多弱裂缝（仅 123 框/91 图）。
PSEUDO_CONF = 0.25   # 默认值；可由 --conf 覆盖
IMGSZ = 640

# 7 档 → A 组 best.pt（训练起点均为 source_pretrain，50 epochs，见 crossdomain.csv）
TIERS = [
    "shot10", "shot20", "shot30", "shot50",
    "img100", "img200", "img500",
]


def labeled_subset(tier: str, seed: str) -> Path:
    """返回该档指定 seed 的标注子集 txt 路径。"""
    for base in ("data/fewshot", "data/fewshot_img"):
        p = ROOT / base / f"{tier}_{seed}.txt"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {tier}_{seed}.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=TIERS,
                    help="只跑指定档位（调试用），默认全 7 档")
    ap.add_argument("--seed", default="seed42", help="抽样种子，默认 seed42")
    ap.add_argument("--conf", type=float, default=PSEUDO_CONF,
                    help=f"伪标签置信阈值 τ，默认 {PSEUDO_CONF}")
    ap.add_argument("--suffix", default="",
                    help="输出目录后缀（如 _t050），用于区分不同 τ，避免覆盖")
    args = ap.parse_args()
    seed = args.seed
    conf = args.conf
    suffix = args.suffix

    train_txt = ROOT / "data" / "splits" / "damcrack_train.txt"
    all_train = [p.strip() for p in train_txt.read_text(encoding="utf-8").splitlines() if p.strip()]

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 需要 ultralytics，请用 anaconda3/python.exe 运行")

    for tier in args.tiers:
        best = ROOT / "runs" / f"pw_yolo_{tier}_{seed}" / "weights" / "best.pt"
        if not best.exists():
            print(f"[SKIP] {tier}: 缺权重 {best}")
            continue

        sub = labeled_subset(tier, seed)
        labeled_names = {Path(p).name for p in sub.read_text(encoding="utf-8").splitlines() if p.strip()}
        pool = [p for p in all_train if Path(p).name not in labeled_names]

        # 防泄漏检查：池里不得含 val/test
        pool_names = {Path(p).name for p in pool}
        for split in ("damcrack_val.txt", "damcrack_test.txt"):
            sp = ROOT / "data" / "splits" / split
            overlap = pool_names & {Path(p).name for p in sp.read_text(encoding="utf-8").splitlines() if p.strip()}
            if overlap:
                raise RuntimeError(f"[LEAK] {tier} 无标注池与 {split} 重叠 {len(overlap)} 张")

        out = RESULTS / "predictions" / f"pseudo_{tier}_{seed}{suffix}"
        print(f"[pseudo] {tier}: 池 {len(pool)} 张, τ={conf}, -> {out.relative_to(ROOT)}")
        model = YOLO(str(best))
        lbl_dir = predict_to_txt(model=model, images=pool, out_dir=out,
                                 conf=conf, imgsz=IMGSZ)
        n_txt = len(list(Path(lbl_dir).glob("*.txt")))
        print(f"[done]  {tier}: 产出 {n_txt} 个预测 txt（其余为无框图，合并时丢弃）")

    print("[ALL DONE] 下一步运行 29_merge_pseudo.py")


if __name__ == "__main__":
    main()
