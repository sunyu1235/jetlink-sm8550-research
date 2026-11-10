# Jetlink Light Speed / Snapdragon 8 Gen 2 研究交接

这是从本地工作区导出的独立研究快照，供 Dot 审查。源码、证据和结论按原样保留；不是已部署的优化版 Jetlink。

先读 [独立研究任务](reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/dot-review-request.txt)，再读 [当前结果及证据](reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/README.md)。

新增：[QNN 2.42 / v73 工具链核实与下一步](reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/toolchain-handoff/README.md)。接收方已取得原始模型和 160 帧合成输入，无需重复提供；当前先核对 Hexagon SDK 5.5.5 / Tools 8.7.06 并验证最小算子编译与加载。

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


## 2026-11-09 新研究源码与证据

新增 compact CPU/native integration、重量化、蒸馏、紧凑 GELU 原型及公开道路输入重建脚本。请从 [研究总览](research/README.md) 查看当前通过项、失败项、未完成项及复现边界。已有历史证据与上游源码保持原样。

注意：10.8 MB 只是两个局部分区的 QNN context，不是完整模型。compact CPU 图连同必需权重约 792.134 MB；完整大小核算见 [审计](research/model-size-audit-20261109.json)。原生夜间和白天各 320 帧及各自暖机后的最后 160 帧检查均已通过，属于主机数值验证。未声称手机速度、32 ms 或部署成功。
