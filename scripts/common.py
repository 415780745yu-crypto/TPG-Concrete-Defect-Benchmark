"""共用工具：配置加载、路径解析、CSV 追加、显式校验。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("[STOP] 缺少 PyYAML。请先在项目根目录执行: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def load_cfg(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "datasets.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def need(cond: bool, msg: str) -> None:
    """任何不确定都停止，绝不猜测数据集结构。"""
    if not cond:
        print(f"[STOP] {msg}", file=sys.stderr)
        sys.exit(2)


def abs_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def append_csv(name: str, row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    write_header = not p.exists()
    with open(p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def read_yolo_label(path: Path) -> list[tuple[int, int]]:
    """返回 [(class_id, n_instances)]；跳过空行。"""
    counts: dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid = int(float(line.split()[0]))
            counts[cid] = counts.get(cid, 0) + 1
    return sorted(counts.items())


def inspect_voc_names(xml_paths: list[Path]) -> list[str]:
    """扫描 VOC XML 中出现的全部类别名，供用户填写 source_class_ids，不猜测类别映射。"""
    import xml.etree.ElementTree as ET

    names: set[str] = set()
    for p in xml_paths:
        root = ET.parse(p).getroot()
        for obj in root.findall("object"):
            n = obj.findtext("name")
            if n:
                names.add(n.strip())
    return sorted(names)


def voc_to_yolo(xml_path: Path, out_dir: Path, name_to_id: dict) -> list[tuple[int, int]]:
    """VOC XML -> YOLO txt（归一化 cx cy w h），返回 [(class_id, 实例数)]。

    仅用于 CUBIT-Det：官方分发的是 VOC XML。DamCrack 自带 YOLO txt，不走这里。
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    need(size is not None, f"{xml_path.name}: VOC XML 缺少 <size>，无法归一化坐标")
    w, h = float(size.findtext("width")), float(size.findtext("height"))
    need(w > 0 and h > 0, f"{xml_path.name}: 图像尺寸非法 {w}x{h}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = defaultdict(int)
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in name_to_id:
            raise SystemExit(
                f"[STOP] VOC 标注中出现未映射类别 '{name}'（文件 {xml_path.name}）。"
                "请把实际类别名补进 configs/datasets.yaml 的 source_class_ids 后重跑。"
            )
        cid = int(name_to_id[name])
        bb = obj.find("bndbox")
        need(bb is not None, f"{xml_path.name}: object 缺少 bndbox")
        x0, y0 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x1, y1 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        lines.append(
            f"{cid} {((x0 + x1) / 2) / w:.6f} {((y0 + y1) / 2) / h:.6f} "
            f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
        )
        counts[cid] += 1
    (out_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sorted(counts.items())


def write_ultralytics_yaml(out: Path, root: Path, splits: dict[str, Path], names: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {root.as_posix()}",
        f"train: {splits['train'].as_posix()}",
        f"val: {splits['val'].as_posix()}",
        f"test: {splits['test'].as_posix()}",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

def predict_to_txt(model, images, out_dir, conf: float, imgsz: int, chunk: int = 200) -> "Path":
    """分块预测并保存 YOLO txt（含 conf），返回保存 txt 的目录。

    Ultralytics predict 对长列表会累积 Results 对象、撑爆 6GB 显存；
    故按 chunk 分块、stream 逐图处理，txt 按 stem 落盘互不冲突。
    Ultralytics save_txt 是追加模式，重复预测会翻倍，故每次先清空输出目录。
    """
    from pathlib import Path as _P
    import shutil
    out_dir = _P(out_dir)
    lbl = out_dir / "labels"
    if lbl.exists():
        shutil.rmtree(lbl)
    for i in range(0, len(images), chunk):
        batch = images[i : i + chunk]
        gen = model.predict(
            source=[str(p) for p in batch], conf=conf, imgsz=imgsz,
            save=False, save_txt=True, save_conf=True, project=str(out_dir),
            name="labels", exist_ok=True, verbose=False, stream=True,
        )
        for _ in gen:
            pass
    cand = out_dir / "labels" / "labels"
    return cand if cand.exists() else out_dir / "labels"
