#!/usr/bin/env python3
"""Export Jetlink research evidence into a standalone review repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PIN = "a863c3801da6fd925d14eb1f5b1e250f8efcc649"
REL = Path("reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/upstream-pinned-source")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("Refusing to overwrite an existing export")
    destination.mkdir(parents=True)
    manifest = []

    def copy(source: Path, relative: Path, provenance: str) -> None:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Expected regular file: {source}")
        if source.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f"File exceeds GitHub normal Git limit: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        before, after = digest(source), digest(target)
        if before != after:
            raise ValueError(f"Copy hash mismatch: {relative}")
        manifest.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                         "sha256": after, "provenance": provenance})

    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "reports"], cwd=ROOT
    ).decode("utf-8").split("\0")
    for name in sorted(set(names)):
        relative = Path(name)
        if len(relative.parts) > 1 and relative.parts[0] == "reports" and relative.parts[1].startswith("jetlink"):
            copy(ROOT / relative, relative, "local Jetlink research evidence")

    upstream = args.upstream.resolve()
    if not (upstream / "LICENSE").is_file():
        raise SystemExit("Pinned Jetlink source/license missing")
    for source in sorted(upstream.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(upstream)
        if any(part in {".git", "__pycache__", "node_modules"} for part in relative.parts):
            continue
        if relative.parts[:2] == ("docs", "images"):
            continue
        copy(source, Path("vendor/jetlink") / relative, f"zoompilot/jetlink@{PIN}")

    (destination / ".gitattributes").write_text("* -text\n*.npy binary\n*.bin binary\n", encoding="utf-8")
    (destination / ".gitignore").write_text("__pycache__/\n*.py[cod]\n.venv/\n/outputs/\n", encoding="utf-8")
    readme = """# Jetlink Light Speed / Snapdragon 8 Gen 2 研究交接

这是从本地工作区导出的独立研究快照，供 Dot 审查。源码、证据和结论按原样保留；不是已部署的优化版 Jetlink。

先读 [独立研究任务](reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/dot-review-request.txt)，再读 [当前结果及证据](reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/README.md)。

关键数据：旧全 NPU 方案约 32.48ms，但数值对照失败；准确修复目前最好 52.41ms，160 帧通过原始检查。混合 NPU/GPU 方案约 65.28ms，160 帧的两个 quiet plan 列未通过。此前没有找到兼顾约 30ms 和精度的实现，但这不证明硬件不可能；请独立审查，寻找有具体依据的新修复。

## 内容与边界

- `reports/jetlink-*`：探针源码、构图脚本、原始日志、CSV、manifest、逐 head 检查记录及已保存到 reports 的 NumPy 对照数组。
- `reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/legacy-32ms/`：旧 32ms 方案的记录、C 探针及 profile；原始 1200 帧运行的 stdout/CSV 未找回。
- `vendor/jetlink/`：上游 zoompilot/jetlink 固定提交 `a863c3801da6fd925d14eb1f5b1e250f8efcc649` 的源码副本；仅去除了演示图片/视频和本地缓存。其 MIT LICENSE 及源码版权声明保留，未修改上游源码。
- `snapshot-manifest.json`：所有导入文件的字节数和 SHA256；Git 禁止换行自动转换，以保持证据字节不变。
- 未包含大型 ONNX/TFLite 权重、QNN 编译缓存、运行库、设备可执行文件或 outputs 内的输入 tensor captures。报告内的旧绝对路径只记录来源，不能从远端访问。
- 原始模型下载地址和哈希在研究任务文件中。此快照没有原工作区的其他项目代码、Git 历史、登录凭据或密钥。

## 直接复核已保存的数值结果

需要 Python 和 NumPy。在仓库根目录执行（Linux/macOS shell）：

```sh
PYTHONPATH=vendor/jetlink python vendor/jetlink/scripts/verify_parity.py compare --dir reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare
```

预期通过。将目录改为 `reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/mixed-litert-160-confirm/compare`，预期失败，原因是两个 quiet plan 列。检查器及阈值保持上游原版。这能复核已保存数组的比较结论，不能重新生成手机输出或完成真实道路精度验证。

旧脚本中可能有本地 Windows/Android 路径。先审查依赖，勿直接执行设备脚本；需要新实验时应先明确新的可证伪假设、缺失工具和数据。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    data = {"upstream_pin": PIN, "source_scope": "Jetlink research reports and pinned upstream source",
            "files": manifest, "file_count": len(manifest), "bytes": sum(x["bytes"] for x in manifest)}
    (destination / "snapshot-manifest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "files": len(manifest), "bytes": data["bytes"]}))


if __name__ == "__main__":
    main()
