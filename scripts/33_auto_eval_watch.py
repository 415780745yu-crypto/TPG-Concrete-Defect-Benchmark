# -*- coding: utf-8 -*-
"""33: 自动评估看门狗 —— 逐档检测训练完成并自动评估，最后重生成报告。

完成判定（不依赖 job 日志，纯文件系统）：
  该档 weights/best.pt 存在，且 weights/last.pt 的 mtime 已停滞超过 --stale-min 分钟。
  （训练中 last.pt 每个 epoch 更新；EarlyStop/正常结束后停止更新。）

幂等：若 crossdomain.csv 已有 <tier>_seed42_pl 行则跳过评估。

用法：
  python scripts/33_auto_eval_watch.py                # 单轮：评估所有已完成的档
  python scripts/33_auto_eval_watch.py --watch        # 循环直到 7 档全部完成
  python scripts/33_auto_eval_watch.py --watch --stale-min 5 --poll-sec 120
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import ROOT, RESULTS

TIERS = ["shot10", "shot20", "shot30", "shot50", "img100", "img200", "img500"]


def run_weights(tier: str) -> Path:
    return ROOT / "runs" / f"pw_yolo_{tier}_seed42_pl" / "weights"


def is_done(tier: str, stale_min: float) -> bool:
    w = run_weights(tier)
    best, last = w / "best.pt", w / "last.pt"
    if not best.exists():
        return False
    if not last.exists():
        return False
    age_min = (time.time() - last.stat().st_mtime) / 60.0
    return age_min > stale_min


def already_evaluated() -> set[str]:
    """返回已写入 crossdomain.csv 的 tag 集合。

    ⚠️ 必须用 csv 模块解析，不能 split(',')：
    PowerShell Export-Csv 会给字段加引号（"tag"），朴素 split 会把引号带进 tag，
    导致比较失败 → 每轮重复评估全部档位（已实测踩坑）。
    """
    import csv as _csv
    p = RESULTS / "crossdomain.csv"
    if not p.exists():
        return set()
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        return {r["tag"].strip() for r in _csv.DictReader(f) if r.get("tag")}


def dedup_crossdomain() -> None:
    """去除 crossdomain.csv 中重复 tag（保留最后一条）。

    04_eval.py 用 append_csv 追加；若同一 tag 被手动+看门狗两次评估，
    会产生数值相同但重复的行，污染结果表。每次评估后调用本函数。
    """
    import csv as _csv
    p = RESULTS / "crossdomain.csv"
    if not p.exists():
        return
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
        fields = rows[0].keys() if rows else []
    if not rows:
        return
    seen: dict[str, dict] = {}
    for r in rows:
        seen[r["tag"]] = r
    if len(seen) == len(rows):
        return
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(seen.values())
    print(f"[dedup] crossdomain.csv {len(rows)} -> {len(seen)} 行")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="循环监控直到 7 档全部完成")
    ap.add_argument("--stale-min", type=float, default=5.0,
                    help="last.pt 停滞多少分钟视为训练结束（默认 5）")
    ap.add_argument("--poll-sec", type=int, default=120, help="监控轮询间隔秒（默认 120）")
    args = ap.parse_args()

    py = sys.executable
    while True:
        done_tags = already_evaluated()
        todo = []
        for t in TIERS:
            tag = f"{t}_seed42_pl"
            if tag in done_tags:
                continue
            if is_done(t, args.stale_min):
                todo.append((t, tag))

        stamp = datetime.now().strftime("%H:%M:%S")
        if todo:
            print(f"[{stamp}] 待评估: {[t for t, _ in todo]}")
            for tier, tag in todo:
                best = run_weights(tier) / "best.pt"
                print(f"[eval] {tag}")
                r = subprocess.run(
                    [py, str(ROOT / "scripts" / "04_eval.py"),
                     "--weights", str(best),
                     "--data", str(ROOT / "data" / "damcrack.yaml"),
                     "--tag", tag, "--protocol", "crossdomain",
                     "--note", "pseudo-label"])
                if r.returncode != 0:
                    print(f"[WARN] {tag} eval exit={r.returncode}")
            dedup_crossdomain()
        else:
            pending = [t for t in TIERS if f"{t}_seed42_pl" not in done_tags]
            print(f"[{stamp}] 无新完成档位；未完成: {pending}")

        # 全 7 档都已评估 -> 重生成报告并退出
        done_tags = already_evaluated()
        remaining = [t for t in TIERS if f"{t}_seed42_pl" not in done_tags]
        if not remaining:
            print("[ALL 7 EVALUATED] 重生成报告")
            subprocess.run([py, str(ROOT / "scripts" / "32_pl_report_v2.py")])
            subprocess.run([py, str(ROOT / "scripts" / "30_pl_report.py")])
            break

        if not args.watch:
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
