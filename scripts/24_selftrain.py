"""24: 伪标签自训练 - 生成伪标签集（步骤2）。复用 results/predictions/pseudo_probe 的预测。"""
from __future__ import annotations
import csv
import shutil
from pathlib import Path
from common import ROOT, RESULTS, load_cfg, need

def main():
    cfg = load_cfg()
    subset_txt = ROOT / "data" / "fewshot_img" / "img500_seed42.txt"
    need(subset_txt.exists(), "缺少 img500_seed42.txt")
    imgs = [Path(p) for p in subset_txt.read_text(encoding="utf-8").splitlines() if p.strip()]
    pred_dir = RESULTS / "predictions" / "pseudo_probe" / "labels" / "labels"
    if not pred_dir.exists():
        pred_dir = RESULTS / "predictions" / "pseudo_probe" / "labels"
    need(pred_dir.exists(), "缺少 pseudo_probe 预测，先跑探测")

    dc = cfg["datasets"]["damcrack"]
    root = Path(dc["root"]) if Path(dc["root"]).is_absolute() else ROOT / dc["root"]
    img_src = root / dc["image_dir"]
    val_txt = ROOT / "data" / "splits" / "damcrack_val.txt"
    test_txt = ROOT / "data" / "splits" / "damcrack_test.txt"
    names = cfg["classes"]["target"]

    rows = []
    for thr, tag in [(0.5, "thr050"), (0.25, "thr025"), (0.1, "thr010")]:
        out_dir = ROOT / "data" / "pseudo" / f"img500_seed42_{tag}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        idir = out_dir / "images"; ldir = out_dir / "labels"
        idir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)
        n_box = 0; n_img = 0
        for im in imgs:
            shutil.copy2(img_src / im.name, idir / im.name)
            txt = pred_dir / f"{im.stem}.txt"
            lines = []
            if txt.exists():
                for line in txt.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) >= 6:
                        cid = int(float(p[0])); conf = float(p[5])
                        if cid in (0, 1) and conf >= thr:
                            lines.append(f"{cid} {p[1]} {p[2]} {p[3]} {p[4]}")
            if lines:
                n_img += 1; n_box += len(lines)
            (ldir / f"{im.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        (out_dir / "train.txt").write_text("\n".join((idir / im.name).as_posix() for im in imgs) + "\n", encoding="utf-8")
        (out_dir / "data.yaml").write_text("\n".join([
            f"path: {out_dir.as_posix()}", "train: train.txt",
            f"val: {val_txt.as_posix()}", f"test: {test_txt.as_posix()}",
            f"nc: {len(names)}", "names:"] + [f"  {k}: {v}" for k, v in sorted(names.items())]) + "\n", encoding="utf-8")
        rows.append({"tag": f"st500_{tag}", "conf_thr": thr, "n_pseudo_boxes": n_box,
                     "images_with_pseudo": n_img, "map50": "", "note": "self-training"})
        print(f"[gen] {tag}: {n_box} 伪框 / {n_img} 图 -> {out_dir.relative_to(ROOT)}")

    with open(RESULTS / "selftrain.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "conf_thr", "n_pseudo_boxes", "images_with_pseudo", "map50", "note"])
        w.writeheader(); w.writerows(rows)
    print("[write] results/selftrain.csv")

if __name__ == "__main__":
    main()
