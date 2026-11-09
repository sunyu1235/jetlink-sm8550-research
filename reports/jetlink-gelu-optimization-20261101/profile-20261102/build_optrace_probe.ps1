$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$report = Join-Path $root 'reports\jetlink-gelu-optimization-20261101\profile-20261102'
$repair = Join-Path $root 'reports\jetlink-gelu-repair-20261101'
$source = Join-Path $report 'history_probe_optrace.c'
$base = Join-Path $repair 'history_probe.c'
$oldPath = '/data/local/tmp/jetlink_qnn_profile_basic.csv'
$newPath = '/data/local/tmp/jetlink_qnn_optrace_20261102.csv'
$sourceText = Get-Content -LiteralPath $base -Raw
if (!$sourceText.Contains($oldPath)) { throw "Expected profile path not found in $base" }
if (Test-Path -LiteralPath $source) { throw "Refusing to overwrite $source" }
[System.IO.File]::WriteAllText($source, $sourceText.Replace($oldPath, $newPath), [System.Text.UTF8Encoding]::new($false))

$clang = 'C:\Users\sunbo\Documents\moto-x40-root\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd'
$ort = Join-Path $env:USERPROFILE 'Documents\moto-x40-root\include'
$binary = Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\profile-optrace\history_probe_optrace'
if (!(Test-Path -LiteralPath $clang)) { throw "NDK compiler missing: $clang" }
if (!(Test-Path -LiteralPath (Join-Path $ort 'onnxruntime\onnxruntime_c_api.h'))) { throw "ORT headers missing: $ort" }
if (Test-Path -LiteralPath $binary) { throw "Refusing to overwrite $binary" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $binary) | Out-Null
& $clang '-O2' '-Wall' '-Wextra' '-Werror' '-DJL_ORT_DLOPEN' "-I$repair" "-I$ort" $source (Join-Path $repair 'qnn_jl_custom.c') '-ldl' '-o' $binary
if ($LASTEXITCODE -ne 0) { throw "Compiler exit $LASTEXITCODE" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant()
@{
  source = $source
  compiler = $clang
  target = 'aarch64-linux-android24'
  flags = @('-O2','-Wall','-Wextra','-Werror','-DJL_ORT_DLOPEN')
  qnn_profile_level = 'optrace'
  qnn_profile_path = $newPath
  binary = $binary
  sha256 = $hash
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report 'optrace-build.json') -Encoding utf8
Write-Output "$binary SHA256=$hash"
