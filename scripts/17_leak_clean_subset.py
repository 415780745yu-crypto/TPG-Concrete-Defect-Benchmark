"""17: 泄漏影响量化 + clean test 子集重评。

从 leak_check.csv 找 CUBIT-Det test 中被 train 近重复命中的唯一图像，生成 clean test 清单，
用 source_pretrain 与 a1_ghostnet 在 clean test 上各评一次（3 类、注明 n_images），
写 results/leak_clean_subset.csv，并给出相对全量 test 的 Δ。

用法：python scripts/17_leak_clean_subset.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from common import ROOT, RESULTS, load_cfg, need

BASELINE = {"source_pretrain": 0.7153, "a1_ghostnet": 0.4603}


def main():
    cfg = load_cfg()
    leak = RESULTS / "leak_check.csv"
    need(leak.exists(), "先跑 16_leak_check.py 生成 leak_check.csv")

    affected = set()
    with open(leak, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["domain"] == "cubit_det" and r["split_a"] == "train" and r["split_b"] == "test":
                affected.add(r["image_b"])

    test_txt = ROOT / "data" / "splits" / "cubit_det_test.txt"
    need(test_txt.exists(), "缺少 cubit_det_test.txt")
    test_imgs = [Path(x) for x in test_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    clean = [p for p in test_imgs if p.name not in affected]
    clean_txt = ROOT / "data" / "splits" / "cubit_det_test_clean.txt"
    clean_txt.write_text("\n".join(p.as_posix() for p in clean) + "\n", encoding="utf-8")

    sc = cfg["datasets"]["cubit_det"]
    root = Path(sc["root"]) if Path(sc["root"]).is_absolute() else ROOT / sc["root"]
    names = cfg["classes"]["source"]
    yaml = ROOT / "data" / "cubit_det_clean.yaml"
    lines = [f"path: {root.as_posix()}",
             f"train: {(ROOT/'data'/'splits'/'cubit_det_train.txt').as_posix()}",
             f"val: {(ROOT/'data'/'splits'/'cubit_det_val.txt').as_posix()}",
             f"test: {clean_txt.as_posix()}",
             f"nc: {len(names)}", "names:"]
    lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[leak] 受影响唯一 test 图像: {len(affected)} / {len(test_imgs)} "
          f"({len(affected)/len(test_imgs)*100:.2f}%)；clean test: {len(clean)} 张")

    from ultralytics import YOLO
    rows = []
    for tag, weights in [("source_only_clean", "runs/pw_yolo_source_pretrain/weights/best.pt"),
                         ("a1_clean", "runs/pw_yolo_a1_ghostnet/weights/best.pt")]:
        model = YOLO(str(ROOT / weights))
        m = model.val(data=str(yaml), split="test", imgsz=cfg["imgsz"], verbose=False)
        box = m.box
        map50 = round(float(box.map50), 4)
        ap50 = {}
        for i, nm in m.names.items():
            try:
                ap50[nm] = round(float(box.ap50[i]), 4)
            except Exception as exc:
                print(f"[warn] {tag} 分类别 AP 未取得: {exc}")
        base = BASELINE["source_pretrain" if "source" in tag else "a1_ghostnet"]
        delta = round(map50 - base, 4)
        rows.append({
            "tag": tag, "weights": str(ROOT / weights), "n_images": len(clean),
            "map50_clean": map50, "map50_full_test": base, "delta_clean_minus_full": delta,
            "ap50_crack": ap50.get("crack", ""), "ap50_spalling": ap50.get("spalling", ""),
            "ap50_moisture": ap50.get("moisture", ""),
        })
        print(f"[eval] {tag}: clean mAP50={map50} (n={len(clean)}) Δ={delta}")

    with open(RESULTS / "leak_clean_subset.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[write] results/leak_clean_subset.csv")

    # 决策规则
    d = abs(rows[0]["delta_clean_minus_full"])
    print(f"\n[decision] source_only_clean Δ = {rows[0]['delta_clean_minus_full']} (|Δ|={d})")
    if d <= 0.02:
        print("[decision] |Δ|≤0.02 → 披露即可，不返工")
    elif d <= 0.05:
        print("[decision] 0.02<|Δ|≤0.05 → 披露 + 主表脚注给干净值、摘要用保守值")
    else:
        print("[decision] |Δ|>0.05 → 停下，等用户确认，禁止自行重划分/重跑源域")


if __name__ == "__main__":
    main()
