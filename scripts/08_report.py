"""阶段 6：把 results/*.csv 汇总成论文表 3 / 5 / 6 与图 5 / 6 的数据源。

脚本只做重塑与透传，不生成任何估算值；缺哪张表会明确打印。
表 5 口径（用户定稿）：
  - shot 档报均值±标准差，图 5 带误差棒；
  - 表 5 加 images 列（每档 3 个 seed 的图像数范围）；
  - source_only 是源域 3 类，其余为目标域 2 类，禁止相减宣称掉点百分比；
  - target_to_source 是双向（T→S）单独成行，不与 forward zero-shot 相减。
方案 B 不报 shots_for_90pct（见 09 脚本），只给定性结论。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, need


def read(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists() or p.stat().st_size == 0:
        print(f"[缺少] results/{name} -> 对应实验尚未跑")
        return None
    return pd.read_csv(p)


def save(df: pd.DataFrame, name: str) -> None:
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False, encoding="utf-8-sig")
    print(f"[生成] results/tables/{name}  ({len(df)} 行)")


def build_table5(cross: pd.DataFrame, fewshot: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cross.copy()
    df["tag"] = df["tag"].astype(str)
    df["protocol"] = df["protocol"].fillna("").astype(str)

    bidi = df[df["protocol"] == "bidirectional"].copy()
    df = df[df["protocol"] != "bidirectional"].copy()

    shot_mask = df["tag"].str.contains(r"shot\d+", regex=True, na=False)
    shots = df[shot_mask].copy()
    shots["shot"] = shots["tag"].str.extract(r"shot(\d+)").astype(int)
    fixed = df[~shot_mask].copy()

    # 每档 images 范围
    images_map = {}
    if fewshot is not None and "shots" in fewshot.columns and "images" in fewshot.columns:
        fi = fewshot.copy()
        fi["shot"] = fi["shots"].astype(int)
        for s, g in fi.groupby("shot"):
            images_map[int(s)] = f"{int(g['images'].min())}–{int(g['images'].max())}"

    def num(col, row):
        v = row.get(col)
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return ""

    rows = []
    # 固定行（非 shot）
    role = {
        "source_only": "源域内参考（源域 3 类）",
        "zero_shot": "泛化下界（目标域 2 类）",
        "full_ft": "全量微调（目标域 2 类）",
        "target_only": "目标域专属上界（目标域 2 类）",
    }
    order_fixed = {"source_only": 0, "zero_shot": 1, "full_ft": 8, "target_only": 9}
    for _, r in fixed.iterrows():
        tag = str(r["tag"])
        rows.append({
            "setting": tag, "shot": np.nan, "map50": num("map50", r), "map50_std": "",
            "ap50_crack": num("ap50_crack", r), "ap50_spalling": num("ap50_spalling", r),
            "images": "", "scope": role.get(tag, "目标域 2 类"), "note": "",
            "_order": order_fixed.get(tag, 5),
        })

    # shot 聚合
    for s, g in shots.groupby("shot"):
        s = int(s)
        rows.append({
            "setting": f"shot{s}", "shot": s,
            "map50": round(float(g["map50"].mean()), 4),
            "map50_std": round(float(g["map50"].std(ddof=0)), 4),
            "ap50_crack": round(float(g["ap50_crack"].mean()), 4) if "ap50_crack" in g and g["ap50_crack"].notna().all() else "",
            "ap50_spalling": round(float(g["ap50_spalling"].mean()), 4) if "ap50_spalling" in g and g["ap50_spalling"].notna().all() else "",
            "images": images_map.get(s, ""),
            "scope": "目标域 2 类",
            "note": "shot=每类 N 个带框实例；3 seeds 均值；seed42 系统性偏低为采样协议效应",
            "_order": 2 + s / 100.0,
        })

    # 双向行
    for _, r in bidi.iterrows():
        rows.append({
            "setting": str(r["tag"]), "shot": np.nan,
            "map50": num("map50", r), "map50_std": "",
            "ap50_crack": num("ap50_crack", r), "ap50_spalling": num("ap50_spalling", r),
            "images": "", "scope": "目标域→源域（2 共享类）",
            "note": "自有 AP 实现、仅 crack/spalling 两个共有类；不与 forward zero-shot 相减",
            "_order": 10,
        })

    table5 = pd.DataFrame(rows).sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # fig5 数据：shot 均值±标准差（带参考线）
    fig5_rows = []
    for _, r in table5[table5["shot"].notna()].iterrows():
        fig5_rows.append({
            "shot": int(r["shot"]), "map50_mean": r["map50"], "map50_std": r["map50_std"],
            "n_seeds": 3,
        })
    refs = {}
    for _, r in table5.iterrows():
        if r["setting"] in ("source_only", "target_only", "zero_shot"):
            refs[r["setting"]] = r["map50"]
    fig5 = pd.DataFrame(fig5_rows)
    fig5.attrs["refs"] = refs
    return table5, fig5


def plot_fig5(fig5: pd.DataFrame, refs: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        x = fig5["shot"].astype(int).tolist()
        y = fig5["map50_mean"].tolist()
        err = fig5["map50_std"].tolist()
        ax.errorbar(x, y, yerr=err, fmt="-o", capsize=4, color="#1f77b4", label="few-shot (3 seeds)")
        for name, color, ls in [("source_only", "#2ca02c", "--"), ("target_only", "#d62728", "--"), ("zero_shot", "#7f7f7f", ":")]:
            if name in refs:
                ax.axhline(refs[name], color=color, linestyle=ls, linewidth=1, label=name)
        ax.set_xlabel("shot (instances per class)")
        ax.set_ylabel("mAP50 (target domain)")
        ax.set_title("Cross-domain few-shot recovery")
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig5_recovery.png", dpi=150)
        print(f"[生成] results/fig5_recovery.png")
    except Exception as exc:
        print(f"[warn] 图 5 未生成: {exc}")


def _std0(s: pd.Series) -> float:
    return float(s.std(ddof=0))


def plot_fig5a(cross: pd.DataFrame) -> None:
    """图 5a：实例级微调曲线（shot 10/20/30/50，3-seed 误差棒，ddof=0）。"""
    shots = cross[cross["tag"].astype(str).str.contains(r"shot\d+", regex=True, na=False)].copy()
    if len(shots) == 0:
        print("[缺少] 图 5a：crossdomain.csv 无 shot 记录")
        return
    shots["shot"] = shots["tag"].astype(str).str.extract(r"shot(\d+)").astype(int)
    g = shots.groupby("shot")["map50"].agg(mean="mean", std=_std0, n="count").reset_index()
    src = float(cross[cross["tag"] == "source_only"]["map50"].iloc[0]) if (cross["tag"] == "source_only").any() else None

    rows = [{"x": int(r["shot"]), "y_mean": round(float(r["mean"]), 4),
             "y_std": round(float(r["std"]), 4), "n_seeds": int(r["n"])} for _, r in g.iterrows()]
    out = RESULTS / "tables" / "fig5a_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[生成] results/tables/fig5a_data.csv  ({len(rows)} 行)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        xs = [r["x"] for r in rows]; ys = [r["y_mean"] for r in rows]; errs = [r["y_std"] for r in rows]
        ax.errorbar(xs, ys, yerr=errs, fmt="-o", capsize=4, color="#1f77b4", label="instance-level (3 seeds)")
        if src is not None:
            ax.axhline(src, color="#7f7f7f", linestyle="--", linewidth=1)
            ax.text(11, src + 0.012, f"source-domain {src:.4f}, different label space", fontsize=8, color="#555555")
        ax.set_xlabel("Labelled instances per class")
        ax.set_ylabel("mAP50 (target domain)")
        ax.set_title("Fig 5a: instance-level few-shot budget")
        ax.set_xticks([10, 20, 30, 50])
        ax.set_ylim(0, 0.75)
        ax.grid(alpha=0.25)
        fig.text(0.01, 0.01,
                 "Caption: 5a X-axis = labelled instances per class (different unit from 5b); "
                 "these 4 points correspond to 5-38 actual images.", fontsize=7)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(RESULTS / "fig5a_instance_budget.png", dpi=150)
        print("[生成] results/fig5a_instance_budget.png")
    except Exception as exc:
        print(f"[warn] 图 5a 未生成: {exc}")


def plot_fig5b(cross: pd.DataFrame) -> None:
    """图 5b：图像级微调曲线（img 100/200/500，误差棒按可用 seed 数）。"""
    imgs = cross[cross["tag"].astype(str).str.startswith("img", na=False)].copy()
    if len(imgs) == 0:
        print("[缺少] 图 5b：crossdomain.csv 无 img 记录")
        return
    imgs["budget"] = imgs["tag"].astype(str).str.extract(r"img(\d+)").astype(int)
    g = imgs.groupby("budget")["map50"].agg(mean="mean", std=_std0, n="count").reset_index()
    tgt = float(cross[cross["tag"] == "target_only"]["map50"].iloc[0]) if (cross["tag"] == "target_only").any() else None

    rows = []
    for _, r in g.iterrows():
        n = int(r["n"])
        rows.append({"x": int(r["budget"]), "y_mean": round(float(r["mean"]), 4),
                     "y_std": round(float(r["std"]), 4) if n >= 2 else "",
                     "n_seeds": n, "note": f"n={n} seed" + ("s" if n > 1 else "")})
    out = RESULTS / "tables" / "fig5b_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[生成] results/tables/fig5b_data.csv  ({len(rows)} 行)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        xs = [r["x"] for r in rows]
        ys = [r["y_mean"] for r in rows]
        errs = [r["y_std"] if r["y_std"] != "" else 0 for r in rows]
        has_err = any(r["y_std"] != "" for r in rows)
        ax.errorbar(xs, ys, yerr=errs if has_err else None, fmt="-o", capsize=4, color="#2ca02c",
                    label="image-level (n seeds)")
        if tgt is not None:
            ax.axhline(tgt, color="#d62728", linestyle="--", linewidth=1)
            ax.text(110, tgt + 0.012, f"target-domain upper bound {tgt:.4f}", fontsize=8, color="#555555")
        ax.set_xlabel("Labelled images")
        ax.set_ylabel("mAP50 (target domain)")
        ax.set_title("Fig 5b: image-level few-shot budget")
        ax.set_xticks([100, 200, 500])
        ax.set_ylim(0, 0.75)
        ax.grid(alpha=0.25)
        fig.text(0.01, 0.01,
                 "Caption: 5b X-axis = number of labelled images (different unit from 5a); "
                 "the two subplots cannot be merged into a single axis.", fontsize=7)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(RESULTS / "fig5b_image_budget.png", dpi=150)
        print("[生成] results/fig5b_image_budget.png")
    except Exception as exc:
        print(f"[warn] 图 5b 未生成: {exc}")


def main() -> None:
    need(RESULTS.exists(), "results/ 目录不存在，请先跑实验")
    ablation = read("ablation.csv")
    cross = read("crossdomain.csv")
    deploy = read("deploy.csv")
    export = read("export_log.csv")
    if export is not None and "format" in export.columns:
        export = export[export["format"].isin(["onnx", "ncnn"])]
    fewshot = read("fewshot_index.csv")

    if ablation is not None:
        save(ablation, "table3_ablation.csv")

    if cross is not None:
        table5, fig5 = build_table5(cross, fewshot)
        save(table5, "table5_crossdomain.csv")
        save(fig5, "fig5_data.csv")
        refs = getattr(fig5, "attrs", {}).get("refs", {})
        plot_fig5(fig5, refs)
        plot_fig5a(cross)
        plot_fig5b(cross)
        print("[表5口径] source_only=源域3类；其余=目标域2类；禁止相减宣称掉点百分比")

    if deploy is not None:
        save(deploy, "table6_panelA_deploy.csv")
        cols = [c for c in ["tag", "platform", "precision", "latency_ms_mean", "fps", "map50"] if c in deploy.columns]
        if cols:
            save(deploy[cols], "fig6_data.csv")

    if deploy is not None:
        d = deploy.copy()
        d["tag"] = d["tag"].astype(str)
        d["precision"] = d["precision"].astype(str)
        panel_b = d[(d["precision"] == "fp32") | d["tag"].str.contains("_int8_aciq1000", na=False)].copy()
        panel_b = panel_b[["tag", "precision", "map50", "size_MB", "latency_ms_mean", "fps"]]
        save(panel_b, "table6_panelB_tradeoff.csv")
    else:
        print("[缺少] 表 6 下半：需要 results/deploy.csv")

    print("\n[检查] 论文主体所需 = 表 3 + 表 5 + 表 6 上半 + 图 5；其余为加分项")
    print("[纪律] 回填时一次性同步摘要第 4 句、结论首句与图 5 / 图 6 数据点")


if __name__ == "__main__":
    main()
