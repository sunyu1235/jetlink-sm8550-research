$ErrorActionPreference='Stop'
$repo=(Get-Location).Path
$threads=Join-Path $repo 'reports\jetlink-gelu-optimization-20261101\threads'
$repair=Join-Path $repo 'reports\jetlink-gelu-repair-20261101'
$ort=Join-Path $env:USERPROFILE 'Documents\moto-x40-root\include'
$clang='C:\mp-sdk\ndk\27.3.13750724\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android30-clang.cmd'
$out=Join-Path $repo 'outputs\jetlink-gelu-optimization-20261101\threads\history_probe_threads'
if(!(Test-Path $clang)){throw "NDK compiler missing: $clang"}
if(!(Test-Path (Join-Path $ort 'onnxruntime\onnxruntime_c_api.h'))){throw "ORT header missing: $ort"}
$args=@('-O2','-Wall','-Wextra','-Werror','-DJL_ORT_DLOPEN',("-I$threads"),("-I$ort"),(Join-Path $threads 'history_probe_threads.c'),(Join-Path $repair 'qnn_jl_custom.c'),'-ldl','-o',$out)
& $clang @args
if($LASTEXITCODE -ne 0){throw "clang exit $LASTEXITCODE"}
Get-FileHash -Algorithm SHA256 -LiteralPath $out | Select-Object Path,Hash
