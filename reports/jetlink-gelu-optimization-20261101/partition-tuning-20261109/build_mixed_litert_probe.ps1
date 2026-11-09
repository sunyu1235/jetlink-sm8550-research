$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$report = Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$repair = Join-Path $root 'reports\jetlink-gelu-repair-20261101'
$ort = Join-Path $env:USERPROFILE 'Documents\moto-x40-root\include'
$outDir = Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$shimDir = Join-Path $outDir 'upstream-pinned-source\JetlinkKit\Sources\CLiteRt'
$clang = 'C:\Users\sunbo\Documents\moto-x40-root\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd'
$binary = Join-Path $outDir 'mixed_litert_probe'
$temporaryBinary = Join-Path $outDir ('mixed_litert_probe.' + [guid]::NewGuid().ToString('N'))
try {
    & $clang '-std=gnu11' '-O2' '-Wall' '-Wextra' '-Werror' '-DJL_ORT_DLOPEN' "-I$repair" "-I$ort" `
        "-I$(Join-Path $shimDir 'include')" "-I$(Join-Path $shimDir 'vendor')" `
        (Join-Path $report 'mixed_litert_probe.c') (Join-Path $report 'qnn_jl_mixed.c') (Join-Path $shimDir 'jl_litert.c') `
        '-ldl' '-pthread' '-lm' '-o' $temporaryBinary
    if ($LASTEXITCODE -ne 0) { throw "Compiler exit $LASTEXITCODE" }
    Move-Item -LiteralPath $temporaryBinary -Destination $binary -Force
} finally {
    if (Test-Path -LiteralPath $temporaryBinary) { Remove-Item -LiteralPath $temporaryBinary }
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant()
@{
    binary = $binary
    sha256 = $hash
    source = Join-Path $report 'mixed_litert_probe.c'
    source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $report 'mixed_litert_probe.c')).Hash.ToLowerInvariant()
    official_litert_shim = Join-Path $shimDir 'jl_litert.c'
    official_litert_shim_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $shimDir 'jl_litert.c')).Hash.ToLowerInvariant()
    flags = @('-std=gnu11','-O2','-Wall','-Wextra','-Werror','-DJL_ORT_DLOPEN')
    purpose = 'Mixed exact cached HTP vision + LiteRT OpenCL GPU policy with GPU-resident recurrent states'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report 'mixed-litert-build-manifest.json') -Encoding utf8
Write-Output "$binary SHA256=$hash"
