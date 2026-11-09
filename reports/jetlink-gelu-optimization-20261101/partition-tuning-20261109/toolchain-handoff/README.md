# QNN 2.42 / HTP v73 工具链交接（2026-11-09）

本轮找到能匿名下载的官方 QNN 2.42 包，并读取了包内的版本说明和 HTP 示例 Makefile。此前“必须先登录才能取得 QNN 2.42”的判断不再成立。只进行了 HTTP Range 小范围读取，没有下载整包，也没有完成算子编译或真机验证。

## 已核实的配套

QNN 包为 `2.42.0.251225`。其 `examples/QNN/OpPackage/HTP/Makefile` 明确给 v73 设置 **Hexagon SDK 5.5.5 + Hexagon Tools 8.7.06**，编译目标为 `-mv73`，并使用 `-mhvx -mhmx -DPREPARE_DISABLED`。包内 `docs/QAIRT-Docs/QNN/general/setup/linux_setup.html` 也列出 v73 对应 SDK 5.5.5；其中另有 8.6.02 未预装的通用提示，不能用该提示替换 Makefile 中明确的 v73 编译器设置。

Android NDK 用于 ARM64 Android 程序，现有探针构建脚本属于这一类。HTP 包还需要 Hexagon 编译器、QuRT 头文件和相应链接依赖；尚未证明普通 LLVM 或 NDK 可以直接替代这条构建链。

## 已验证可访问的官方下载

- [QNN / QAIRT 2.42.0.251225](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip)：1,543,955,191 字节。
- [Hexagon SDK 6.4.0.2](https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip)：3,090,279,353 字节。
- [Hexagon Tools 19.0.02](https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_open_access/Linux/Debian/19.0.02/Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz)：1,134,735,943 字节。

三条链接均在不带账号或令牌的请求中返回 `206 Partial Content`，前八字节符合 ZIP/GZIP 文件头。6.4.0.2 和 19.0.02 的地址来自 [Qualcomm Hexagon-MLIR 官方安装说明](https://github.com/qualcomm/hexagon-mlir/blob/main/docs/user-guide.md)。这两包可取得，但它们不是上述 QNN 2.42 Makefile 列出的 v73 配套；其与该 legacy OpPackage 的组合仍未验证。没有找到并验证 SDK 5.5.5 / Tools 8.7.06 的匿名直链，不提供猜测地址。

完整压缩包均未下载，所以 `manifest.json` 不填写整包 SHA256。它记录的是 HTTP 探测结果及已按需读取的小文件 SHA256。

## 仓库可放什么

QNN 包内 `LICENSE.pdf` 第一条包含：

> nothing herein grants You a license to distribute or sublicense the Software on a standalone basis.

该条还将分发授权限定在集成到应用中的目标代码。因此不把完整 SDK、工具链、厂商头文件或示例源码独立镜像到这个公开研究仓库。许可证原文件 SHA256 为 `ec1dccfdcba5c6e64126e84199b8362bf4999107bfa567ebe831dbb4c461692b`；本仓库只保存该标识、这段短引文和自写说明。

## 下一步的最小可执行路线

在有足够空间的 Linux 构建环境或 GitHub Actions 的临时构建目录下载官方依赖；SDK 不作为仓库文件、缓存或下载产物再次公开。先取得上述明确配套的 SDK 5.5.5 / Tools 8.7.06，或针对可公开下载的新工具链单独验证兼容性。然后只构建最小 FP16 identity/Relu 包及所需 host/HTP 目标，证明编译、加载和输入输出类型可用，再做 GELU 内核。

原示例 Makefile 还有一个实际障碍：即使只请求 `htp_v73`，它也会做部分全局 SDK 目录检查，并要求较新的 x86 `libnative` 工具路径。不能只设置一个 `HEXAGON_SDK_ROOT` 后假定原 Makefile 会工作。应在临时目录按实际需要调整构建目标和依赖检查，保留厂商原文件，不将其全文提交。

若无法取得旧配套、也不能证明新工具链兼容，就停在“算子构建链尚未打通”。目前没有编译成功、加载成功或因此低于 52.4 ms 的新证据；下载成功不会证明性能收益。
