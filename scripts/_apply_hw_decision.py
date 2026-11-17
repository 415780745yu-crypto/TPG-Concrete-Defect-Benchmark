"""一次性脚本：把端侧硬件基线从「必须树莓派」改为「x86 CPU 为主、ARM 可选、不列功耗」。

运行后打印每条替换的命中数与剩余含旧措辞的行，便于核对。
"""
from pathlib import Path

BASE = Path(r"C:\Users\zhenbai\Desktop\Pytorch\PW-YOLO")
SK = Path(r"C:\Users\zhenbai\WorkBuddy\2026-11-07-16-36-54\output\paper2-imrad-skeleton")

CN = [
    ("PyTorch → ONNX → PNNX → NCNN → x86 CPU / 树莓派 4B 或 5",
     "PyTorch → ONNX → PNNX → NCNN → x86 CPU（主平台；ARM 单板机为可选补充）"),
    ("在树莓派「待实测」上", "在 x86 CPU 上"),
    ("在 x86 CPU 与树莓派 4B/5 上实测 FP32 与 INT8 下的真实延迟、帧率、内存峰值与整机功耗",
     "在 x86 CPU 上实测 FP32 与 INT8 下的真实延迟、帧率与内存峰值"),
    ("→ x86 CPU / 树莓派 4B/5", "→ x86 CPU（ARM 平台可选）"),
    ("> 功耗测量的可操作性：树莓派 4B/5 无板载功率寄存器可读，须外接 USB 功率计或电源管理芯片监测，方法写入 §4.1；若无测量设备则删除该列，不得写「实测」。",
     "> 功耗列：本研究未配备外接 USB 功率计，表 6 与图 8 不列整机功耗；若后续具备设备再补充，禁止在无测量数据时写「实测」。"),
    ("{PyTorch FP32 桌面基准, NCNN FP32 x86, NCNN FP32 树莓派, NCNN INT8 树莓派}",
     "{PyTorch FP32 桌面基准, NCNN FP32 x86, NCNN INT8 x86}"),
    ("列含模型体积、平均延迟、帧率、内存峰值、整机功耗、**量化后 mAP**",
     "列含模型体积、平均延迟、帧率、内存峰值、**量化后 mAP**（无功率计，不列功耗）"),
    ("x86 CPU 与树莓派 4B/5 上的延迟 / 帧率 / 内存 / 功耗对比 + 量化后 mAP",
     "x86 CPU 上 FP32 / INT8 的延迟、帧率、内存对比 + 量化后 mAP"),
    ("§4.5；功耗列须注明测量设备", "§4.5；无功率计时删除功耗列"),
    ("{PyTorch FP32、NCNN FP32 x86、NCNN FP32 树莓派、NCNN INT8 树莓派} × {体积 / 延迟 / 帧率 / 内存 / 功耗 / 量化后 mAP}",
     "{PyTorch FP32、NCNN FP32 x86、NCNN INT8 x86} × {体积 / 延迟 / 帧率 / 内存 / 量化后 mAP}"),
    ("NCNN FP32 / INT8，x86 CPU 与树莓派 4B/5", "NCNN FP32 / INT8，x86 CPU"),
    ("表 6 上半（延迟 / 帧率 / 功耗 / 量化后 mAP）", "表 6 上半（延迟 / 帧率 / 量化后 mAP）"),
    ("- 待核实 是否具备树莓派 4B/5、外接功率计与 INT8 校准所需工具链。",
     "- 待核实 INT8 校准工具链是否可用；ARM 单板机与外接功率计为可选补充，不影响主实验。"),
    ("无嵌入式 GPU 平台实测（如实写「受实验条件限制」，不要包装成「主动不选高功耗板卡」）",
     "端侧实测仅在 x86 CPU 完成，未覆盖 ARM 单板机与嵌入式 GPU（如实写「受实验条件限制」）"),
    ("on Raspberry Pi [TBD]", "on an x86 CPU [TBD]"),
    ("on an x86 CPU and a Raspberry Pi 4B/5 under FP32 and INT8",
     "on an x86 CPU under FP32 and INT8"),
    ("latency, FPS, peak memory and device-level power", "latency, FPS and peak memory"),
    ("on Raspberry Pi [TBD].", "on an x86 CPU [TBD]."),
]

EN = [
    ("ONNX → PNNX → NCNN → x86 CPU / Raspberry Pi 4B/5",
     "ONNX → PNNX → NCNN → x86 CPU (primary; ARM boards optional)"),
    ("on Raspberry Pi [TBD]", "on an x86 CPU [TBD]"),
    ("on an x86 CPU and a Raspberry Pi 4B/5 under FP32 and INT8",
     "on an x86 CPU under FP32 and INT8"),
    ("latency, FPS, peak memory and device-level power", "latency, FPS and peak memory"),
    ("x86 CPU and Raspberry Pi 4B/5", "x86 CPU (ARM boards optional)"),
    ("Raspberry Pi 4B/5", "x86 CPU (ARM boards optional)"),
    ("device-level power", "peak memory"),
    ("whole-device power", "peak memory"),
]


def apply(path: Path, repl: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    print(f"\n=== {path.name} ===")
    for old, new in repl:
        n = text.count(old)
        print(f"{'HIT ' if n else 'MISS'} {n}x  {old[:62]}")
        if n:
            text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        if any(k in line for k in ("树莓派", "整机功耗", "Raspberry Pi", "device-level power")):
            print(f"  [remain L{i}] {line[:120]}")


def prepend_note(path: Path, note: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "## 1. 落地总原则"
    if "硬件基线覆盖说明" not in text and marker in text:
        text = text.replace(marker, note + "\n\n" + marker, 1)
        path.write_text(text, encoding="utf-8")
        print(f"[note inserted] {path.name}")
    else:
        print(f"[note skipped] {path.name}")


if __name__ == "__main__":
    apply(SK / "PW-YOLO_IMRaD_skeleton_中文版.md", CN)
    apply(SK / "PW-YOLO_IMRaD_skeleton.md", EN)
    prepend_note(
        BASE / "落地执行手册.md",
        "## 硬件基线覆盖说明（2026-11-08，优先级最高）\n\n"
        "> 见同目录《方法决策备忘》第 3 节：**端侧实测以本机 x86 CPU 为主平台，树莓派 / ARM 单板机为可选补充，"
        "没有外接功率计则不列整机功耗**。手册下文中凡出现「树莓派 4B/5 实测」「功耗列」的表述，一律按此覆盖；"
        "论文 §3.6 / §4.5 / §5.4 相应改写为「受实验条件限制，端侧实测在 x86 CPU 完成，ARM 与嵌入式 GPU 平台作为未来工作」。",
    )
