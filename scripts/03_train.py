"""阶段 2–4：封装 Ultralytics 训练，并把每次运行的元信息落到 CSV。

典型用法：
  # 源域预训练（S）
  python scripts/03_train.py --data data/cubit_det.yaml --tag source_pretrain
  # 目标域全量微调 / 仅目标域训练
  python scripts/03_train.py --data data/damcrack.yaml --weights runs/pw_yolo/source_pretrain/weights/best.pt --tag full_ft
  python scripts/03_train.py --data data/damcrack.yaml --tag target_only
  # few-shot（N=10, seed=42）
  python scripts/03_train.py --data data/fewshot/shot10_seed42.yaml --weights <source_best.pt> --tag shot10_seed42
  # 显式指定训练种子（不填则取 configs/datasets.yaml 的 seed，当前为 42）
  # 注意区分两个 seed：
  #   - 抽样种子：由 02_fewshot.py 决定"抽哪几张图"，体现在 data/fewshot*/xxx.yaml 里
  #   - 训练种子：本脚本的 --seed，决定增强顺序、dataloader 打乱等
  # 要做"同一批图 × 多个训练种子"，用同一个 --data 分别跑 --seed 42/43/44
  python scripts/03_train.py --data data/fewshot_img/img100_seed42.yaml --tag img100_seed42_tr43 --seed 43
  # 断点续训（中断/关机后从上一轮继续；last.pt 每个 epoch 自动保存）
  python scripts/03_train.py --data data/cubit_det.yaml --tag source_pretrain --resume
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, append_csv, load_cfg, need


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ultralytics data.yaml 路径")
    ap.add_argument("--weights", default=None, help="不填则用 baseline_weights")
    ap.add_argument("--tag", required=True, help="写入 CSV 的实验标签，如 shot10_seed42")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="训练种子；不填则取 configs/datasets.yaml 的 seed（42）。"
                         "注意：数据子集的抽样种子由 02_fewshot.py 决定，与此参数无关")
    ap.add_argument("--model-yaml", default=None, help="消融实验用的自定义结构 yaml，如 configs/pw_yolo_a2.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="从 runs/<name>/weights/last.pt 断点续训；忽略 --weights/--epochs，其余参数沿用 run 目录里的 args.yaml")
    args = ap.parse_args()

    cfg = load_cfg()
    tr = cfg["train"]
    use_seed = cfg["seed"] if args.seed is None else args.seed
    seed_src = "config" if args.seed is None else "CLI"
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    need(data.exists(), f"data.yaml 不存在: {data}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[STOP] 请先 pip install -r requirements.txt")

    weights = args.weights or (args.model_yaml or cfg["baseline_weights"])
    name = f"{tr['name_prefix']}_{args.tag}"

    if args.resume:
        last = ROOT / tr["project"] / name / "weights" / "last.pt"
        need(last.exists(), f"未找到 {last}，没有可恢复的进度；请直接开新训练（不带 --resume）")
        print(f"[train] resume tag={args.tag} from {last}（seed 沿用该 run 的 args.yaml，--seed 不生效）")
        model = YOLO(str(last))
        model.train(resume=True)  # resume 模式下 ultralytics 自动沿用 run 目录的 args.yaml
    else:
        model = YOLO(weights)
        print(f"[train] tag={args.tag} weights={weights} data={data} name={name} "
              f"seed={use_seed}({seed_src})")
        model.train(
            data=str(data),
            epochs=args.epochs or tr["epochs"],
            batch=tr["batch"],
            imgsz=cfg["imgsz"],
            patience=tr["patience"],
            device=args.device or tr["device"],
            workers=tr["workers"],
            seed=use_seed,
            project=str(ROOT / tr["project"]),
            name=name,
            exist_ok=True,
        )
    best = ROOT / tr["project"] / name / "weights" / "best.pt"
    need(best.exists(), f"训练结束但未找到 {best}，请检查 Ultralytics 日志")
    seed_record = "resume(见该 run 的 args.yaml)" if args.resume else use_seed
    append_csv("train_log.csv", {
        "tag": args.tag, "data": str(data), "weights": str(weights),
        "model_yaml": args.model_yaml or "", "epochs": args.epochs or tr["epochs"],
        "seed": seed_record, "best_pt": str(best), "imgsz": cfg["imgsz"],
    })
    print(f"[done] {best}")


if __name__ == "__main__":
    main()
