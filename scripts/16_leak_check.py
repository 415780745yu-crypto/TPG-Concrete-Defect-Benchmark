"""泄漏抽查（16）：两域全部跨 split 图像对做 dHash 汉明距离 ≤8 判近重复。

规格：对每个域（cubit_det / damcrack）的 train/val/test 两两交叉，计算 64-bit dHash，
汉明距离 ≤8 记为近重复。产出：
  results/leak_check.csv        近重复对明细
  results/leak_summary.csv      每域每交叉对的近重复数 / 总对数 / 比率
  results/leak_check_samples.png 抽样对比图（最多 24 对）
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from common import ROOT, RESULTS, load_cfg, need

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def read_split(txt: Path):
    return [Path(x) for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]


def dhash_uint64(cv2, p: Path) -> int:
    img = cv2.imread(str(p))
    if img is None:
        buf = np.fromfile(str(p), dtype=np.uint8)
        need(buf.size > 0, f"无法读取图像 {p}")
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    need(img is not None, f"图像解码失败 {p}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1).astype(np.uint8)
    return int(np.packbits(bits, bitorder="little").view(np.uint64)[0])


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = (a[:, None] ^ b[None, :]).view(np.uint8).reshape(a.shape[0], b.shape[0], 8)
    return POPCOUNT[xor].sum(axis=2)                # (na, nb) int32


def main():
    cfg = load_cfg()
    import cv2
    domains = [("cubit_det", "cubit_det"), ("damcrack", "damcrack")]

    pair_rows, summary_rows, sample_pairs = [], [], []
    for label, key in domains:
        splits = {}
        for sp in ("train", "val", "test"):
            txt = ROOT / "data" / "splits" / f"{key}_{sp}.txt"
            need(txt.exists(), f"缺少 {txt}")
            splits[sp] = read_split(txt)

        hashes = {}
        for sp, imgs in splits.items():
            hs = np.empty(len(imgs), dtype=np.uint64)
            t0 = time.time()
            for i, p in enumerate(imgs):
                hs[i] = dhash_uint64(cv2, p)
            hashes[sp] = (imgs, hs)
            print(f"[dhash] {label}/{sp}: {len(imgs)} 张, {time.time()-t0:.1f}s", flush=True)

        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            imgs_a, ha = hashes[a]
            imgs_b, hb = hashes[b]
            hm = hamming_matrix(ha, hb)
            idx = np.argwhere(hm <= 8)
            total = int(hm.size)
            for i, j in idx:
                pair_rows.append({
                    "domain": label, "split_a": a, "split_b": b,
                    "image_a": imgs_a[int(i)].name, "image_b": imgs_b[int(j)].name,
                    "hamming": int(hm[i, j]),
                })
                if len(sample_pairs) < 24:
                    sample_pairs.append((label, imgs_a[int(i)], imgs_b[int(j)], int(hm[i, j])))
            summary_rows.append({
                "domain": label, "split_a": a, "split_b": b,
                "n_near_dup": int(len(idx)), "n_pairs": total,
                "near_dup_rate": round(len(idx) / total, 6) if total else 0,
            })
            print(f"[check] {label} {a}x{b}: {len(idx)}/{total} 近重复", flush=True)

    with open(RESULTS / "leak_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = ["domain", "split_a", "split_b", "image_a", "image_b", "hamming"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(pair_rows)
    print(f"[write] results/leak_check.csv ({len(pair_rows)} 对)")

    with open(RESULTS / "leak_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = ["domain", "split_a", "split_b", "n_near_dup", "n_pairs", "near_dup_rate"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary_rows)
    print(f"[write] results/leak_summary.csv ({len(summary_rows)} 行)")

    if sample_pairs:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(sample_pairs)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
            axes = np.array(axes).reshape(-1)
            for k, (label, pa, pb, hd) in enumerate(sample_pairs):
                ax = axes[k]
                ima = cv2.imread(str(pa)); imb = cv2.imread(str(pb))
                ima = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB) if ima is not None else np.zeros((10,10,3), np.uint8)
                imb = cv2.cvtColor(imb, cv2.COLOR_BGR2RGB) if imb is not None else np.zeros((10,10,3), np.uint8)
                ax.imshow(np.hstack([ima, imb]))
                ax.set_title(f"{label} h={hd}\n{pa.name} / {pb.name}", fontsize=7)
                ax.axis("off")
            for k in range(n, len(axes)):
                axes[k].axis("off")
            fig.tight_layout()
            fig.savefig(RESULTS / "leak_check_samples.png", dpi=120)
            print("[write] results/leak_check_samples.png")
        except Exception as exc:
            print(f"[warn] 抽样图未生成: {exc}")

    print("[done] 16_leak_check 完成")


if __name__ == "__main__":
    main()
