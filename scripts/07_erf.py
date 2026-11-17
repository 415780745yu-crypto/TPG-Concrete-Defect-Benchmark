"""可选补充材料（骨架 §4.7 / 图 7c）：有效感受野 ERF 的最小实现。

原理：对指定中间层输出特征图的中心点求「相对于输入图像」的梯度，多张图像累加 |梯度| 平均，
得到输入平面上每个像素对该中心响应的贡献分布（Luo et al., NeurIPS 2016）。

用法：
  python scripts/07_erf.py --weights <best.pt> --layer model.4 --tag c2f_baseline
  python scripts/07_erf.py --weights <best.pt> --layer model.4 --tag pconv_layer

先用 `python -c "from ultralytics import YOLO; print(list(YOLO('yolov8n.pt').model.named_modules())[:5])"`
查看层名。对比 C2f 与 PConv 时，必须保证两者处于**相同网络深度、相同初始化状态、相同输入分布**。

重要：该结果是结构归纳偏置的定性观察，只能写「与假说 H1 一致」，不能写「证明」，
也不能单独用来解释 mAP 提升（需另做不同走向裂缝子集 AP 分析）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, append_csv, load_cfg, need


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layer", required=True, help="named_modules 中的层名，如 model.4")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--images", default=None, help="图像目录；不填则用固定种子的随机输入")
    ap.add_argument("--num", type=int, default=32)
    args = ap.parse_args()

    cfg = load_cfg()
    w = Path(args.weights)
    w = w if w.is_absolute() else ROOT / w
    need(w.exists(), f"权重不存在: {w}")

    try:
        import numpy as np
        import torch
        from PIL import Image
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 请先 pip install -r requirements.txt")

    torch.manual_seed(cfg["seed"])
    net = YOLO(str(w)).model.eval()
    target = dict(net.named_modules()).get(args.layer)
    need(target is not None, f"找不到层 {args.layer}，请打印 named_modules 核对")

    holder = {}

    def hook(_module, _inp, out):
        holder["f"] = out[0] if isinstance(out, (list, tuple)) else out

    handle = target.register_forward_hook(hook)

    imgs = []
    if args.images:
        d = Path(args.images)
        d = d if d.is_absolute() else ROOT / d
        files = sorted(d.glob("*"))[: args.num]
        need(len(files) > 0, f"图像目录为空: {d}")
        for f in files:
            im = Image.open(f).convert("RGB").resize((cfg["imgsz"], cfg["imgsz"]))
            imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255.0))
    else:
        imgs = [torch.rand(3, cfg["imgsz"], cfg["imgsz"]) for _ in range(args.num)]

    acc = np.zeros((cfg["imgsz"], cfg["imgsz"]), dtype=np.float64)
    ok = 0
    for im in imgs:
        x = im.unsqueeze(0).clone().requires_grad_(True)
        try:
            with torch.enable_grad():
                net(x)
                f = holder["f"]
                need(torch.is_tensor(f), "hook 抓到的不是 tensor，请换一个卷积层")
                cy, cx = f.shape[-2] // 2, f.shape[-1] // 2
                scalar = f[0, :, cy, cx].abs().sum()
                g = torch.autograd.grad(scalar, x, retain_graph=False)[0]
            acc += g.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
            ok += 1
        except RuntimeError as exc:
            print(f"[warn] 跳过一张图：{exc}")
    handle.remove()
    need(ok > 0, "所有图像都未成功求梯度，请检查层名与网络是否支持 autograd")
    acc /= ok

    center_ratio = float(acc[cfg["imgsz"] // 2 - 16: cfg["imgsz"] // 2 + 16,
                             cfg["imgsz"] // 2 - 16: cfg["imgsz"] // 2 + 16].sum() / acc.sum())
    out_dir = ROOT / "results" / "erf"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.tag}.npy", acc.astype(np.float32))
    try:
        import matplotlib.pyplot as plt
        plt.imsave(out_dir / f"{args.tag}.png", acc / (acc.max() + 1e-12), cmap="viridis")
    except Exception as exc:
        print(f"[warn] 热力图未保存: {exc}")

    append_csv("erf_log.csv", {
        "tag": args.tag, "layer": args.layer, "weights": str(w),
        "num_images": ok, "center_energy_ratio_32x32": round(center_ratio, 4),
        "input_mode": "real" if args.images else "random_fixed_seed",
        "note": "supplementary observation only; consistent with H1, not proof",
    })
    print(f"[ok] {args.tag}: 中心 32x32 能量占比 {center_ratio:.4f} -> results/erf/{args.tag}.png")
    print("[提醒] 报告中心能量占比与衰减速率即可；方向各向异性只作观察量，不预设为优势")


if __name__ == "__main__":
    main()
