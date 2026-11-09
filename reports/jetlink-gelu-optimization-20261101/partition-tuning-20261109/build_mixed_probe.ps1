$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$report = Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$repair = Join-Path $root 'reports\jetlink-gelu-repair-20261101'
$ort = Join-Path $env:USERPROFILE 'Documents\moto-x40-root\include'
$clang = 'C:\Users\sunbo\Documents\moto-x40-root\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd'
$outDir = Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$binary = Join-Path $outDir 'mixed_probe'
$temporaryBinary = Join-Path $outDir ('mixed_probe.' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
try {
    & $clang '-O2' '-Wall' '-Wextra' '-Werror' '-DJL_ORT_DLOPEN' "-I$repair" "-I$ort" `
        (Join-Path $report 'mixed_probe.c') (Join-Path $report 'qnn_jl_mixed.c') '-ldl' '-lm' '-o' $temporaryBinary
    if ($LASTEXITCODE -ne 0) { throw "Compiler exit $LASTEXITCODE" }
    Move-Item -LiteralPath $temporaryBinary -Destination $binary -Force
} finally {
    if (Test-Path -LiteralPath $temporaryBinary) { Remove-Item -LiteralPath $temporaryBinary }
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant()
@{
    sources = @((Join-Path $report 'mixed_probe.c'), (Join-Path $report 'qnn_jl_mixed.c'))
    compiler = $clang
    target = 'aarch64-linux-android24'
    flags = @('-O2','-Wall','-Wextra','-Werror','-DJL_ORT_DLOPEN')
    binary = $binary
    sha256 = $hash
    gpu_binding_memory = 'CPU, selected per session; never QnnHtpShared'
    vision_unused_feature_history = 'zero: no feature/image dependency; actual history owned by GPU policy'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report 'mixed-build-manifest.json') -Encoding utf8
Write-Output "$binary SHA256=$hash"
