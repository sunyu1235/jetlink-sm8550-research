$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$report = Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$outDir = Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$shimDir = Join-Path $outDir 'upstream-pinned-source\JetlinkKit\Sources\CLiteRt'
$clang = 'C:\Users\sunbo\Documents\moto-x40-root\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd'
$source = Join-Path $report 'litert_history_probe.c'
$shim = Join-Path $shimDir 'jl_litert.c'
$binary = Join-Path $outDir 'litert_history_probe'
$tempBinary = Join-Path $outDir ('litert_history_probe.' + [guid]::NewGuid().ToString('N'))
foreach ($required in @($clang, $source, $shim, (Join-Path $shimDir 'include\jl_litert.h'))) {
    if (!(Test-Path -LiteralPath $required)) { throw "Missing build input: $required" }
}
$shimHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $shim).Hash.ToLowerInvariant()
try {
    & $clang '-std=gnu11' '-O2' '-Wall' '-Wextra' '-Werror' "-I$(Join-Path $shimDir 'include')" `
        "-I$(Join-Path $shimDir 'vendor')" $source $shim '-ldl' '-pthread' '-lm' '-o' $tempBinary
    if ($LASTEXITCODE -ne 0) { throw "Compiler exit $LASTEXITCODE" }
    Move-Item -LiteralPath $tempBinary -Destination $binary -Force
} finally {
    if (Test-Path -LiteralPath $tempBinary) { Remove-Item -LiteralPath $tempBinary }
}
$shimHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $shim).Hash.ToLowerInvariant()
if ($shimHashBefore -ne $shimHashAfter) { throw 'Official LiteRT shim changed during build' }
@{
    source = $source
    source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    official_shim = $shim
    official_shim_sha256 = $shimHashAfter
    official_shim_modified = $false
    compiler = $clang
    target = 'aarch64-linux-android24'
    flags = @('-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-ldl', '-pthread', '-lm')
    binary = $binary
    binary_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant()
    purpose = 'Standalone official LiteRT GPU synthetic recurrent replay; no phone execution during build'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report 'litert-build-manifest.json') -Encoding utf8
Get-Item -LiteralPath $binary | Select-Object FullName, Length
Get-FileHash -Algorithm SHA256 -LiteralPath $binary | Select-Object Hash
