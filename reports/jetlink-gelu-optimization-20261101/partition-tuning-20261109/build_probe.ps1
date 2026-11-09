$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$report = Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$repair = Join-Path $root 'reports\jetlink-gelu-repair-20261101'
$ort = Join-Path $env:USERPROFILE 'Documents\moto-x40-root\include'
$clang = 'C:\Users\sunbo\Documents\moto-x40-root\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd'
$outDir = Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$binary = Join-Path $outDir 'partition_probe'
$tempBinary = Join-Path $outDir ('partition_probe.' + [guid]::NewGuid().ToString('N'))
if (!(Test-Path -LiteralPath $clang)) { throw "NDK compiler missing: $clang" }
if (!(Test-Path -LiteralPath (Join-Path $ort 'onnxruntime\onnxruntime_c_api.h'))) { throw "ORT headers missing: $ort" }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
try {
    & $clang '-O2' '-Wall' '-Wextra' '-Werror' '-DJL_ORT_DLOPEN' "-I$repair" "-I$ort" `
        (Join-Path $report 'partition_probe.c') `
        (Join-Path $repair 'qnn_jl_custom.c') '-ldl' '-o' $tempBinary
    if ($LASTEXITCODE -ne 0) { throw "Compiler exit $LASTEXITCODE" }
    Move-Item -LiteralPath $tempBinary -Destination $binary -Force
} finally {
    if (Test-Path -LiteralPath $tempBinary) { Remove-Item -LiteralPath $tempBinary }
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant()
@{
    source = Join-Path $report 'partition_probe.c'
    compiler = $clang
    target = 'aarch64-linux-android24'
    flags = @('-O2', '-Wall', '-Wextra', '-Werror', '-DJL_ORT_DLOPEN')
    binary = $binary
    sha256 = $hash
    runtime_options = @('soc_model via JL_SOC_MODEL', 'vtcm_mb via JL_VTCM_MB')
    qnn_profile_path_default = '/data/local/tmp/jetlink_qnn_profile_runtime_tuning.csv'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report 'build-manifest.json') -Encoding utf8
Write-Output "$binary SHA256=$hash"
