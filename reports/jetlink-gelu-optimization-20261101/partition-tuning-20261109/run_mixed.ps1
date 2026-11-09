param([string]$Serial='ZY22GVVTVZ', [string]$RunId=('mixed-'+(Get-Date -Format 'yyyyMMdd-HHmmss')), [int]$Frames=16)
$ErrorActionPreference='Stop'
$adb='C:\mp-sdk\platform-tools\adb.exe'
$root=(Get-Location).Path
$report=Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$models=Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$logs=Join-Path $report $RunId
$raw=Join-Path $models $RunId
$remote="/data/local/tmp/jetlink-mixed-20261109/$RunId"
if($RunId -notmatch '^[a-zA-Z0-9_-]+$' -or $Frames -lt 1 -or $Frames -gt 160){throw 'Invalid run identifier/frame count'}
if((Test-Path -LiteralPath $logs) -or (Test-Path -LiteralPath $raw)){throw 'Refusing to overwrite a run'}
function Adb([string[]]$AdbArgs){
 $v=& $adb -s $Serial @AdbArgs 2>&1
 if($LASTEXITCODE -ne 0){throw "ADB failed: $($AdbArgs -join ' ')`n$($v -join "`n")"}
 return ($v -join "`n").Trim()
}
Adb @('get-state') | Out-Null
$apk=Adb @('shell','pm path io.zoompilot.jetlink.android')
$baseApk=@($apk -split "`r?`n" | Where-Object {$_ -match 'base.apk$'})[0] -replace '^package:',''
if(!$baseApk -or $baseApk -notmatch '^/data/app/[a-zA-Z0-9_~+=./-]+/base.apk$'){throw 'Unexpected package path'}
$lib=$baseApk -replace '/base.apk$','/lib/arm64'
Adb @('shell',"test ! -e $remote && mkdir -p $remote/shared-boundaries-out") | Out-Null
New-Item -ItemType Directory -Path $logs,$raw | Out-Null
foreach($file in @('mixed_probe','mixed-vision-context.onnx','mixed-policy-fp32-tanh.onnx')){
 $p=Join-Path $models $file
 $target=if($file -eq 'mixed-vision-context.onnx'){"/data/local/tmp/$file"}else{"/data/local/tmp/jetlink-mixed-20261109/$file"}
 $localHash=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
 $old=Adb @('shell',"if test -f $target; then sha256sum $target; else echo missing; fi")
 if($old.Split(' ')[0] -ne $localHash){Adb @('push',$p,$target) | Out-Null}
 $hash=(Adb @('shell',"sha256sum $target")).Split(' ')[0]
 if($hash -ne $localHash){throw "Hash mismatch: $file"}
 "$file $localHash" | Add-Content -LiteralPath (Join-Path $logs 'hashes.txt') -Encoding ascii
}
Adb @('shell','chmod 755 /data/local/tmp/jetlink-mixed-20261109/mixed_probe') | Out-Null
(Adb @('shell','su -c ''cat /sys/class/kgsl/kgsl-3d0/temp''')) | Set-Content -LiteralPath (Join-Path $logs 'shared-boundaries.gpu-before.txt') -Encoding ascii
(Adb @('shell','dumpsys battery; cat /proc/uptime')) | Set-Content -LiteralPath (Join-Path $logs 'before.txt') -Encoding utf8
$cmd="taskset f8 /system/bin/sh -c 'export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:${lib}:/vendor/lib64; export ADSP_LIBRARY_PATH=$lib; export JL_CUSTOM_OP_LIBRARY=/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so; unset JL_ORT_PROFILE JL_CONTEXT_OUTPUT JL_SOC_MODEL JL_VTCM_MB; /data/local/tmp/jetlink-mixed-20261109/mixed_probe /data/local/tmp/mixed-vision-context.onnx /data/local/tmp/jetlink-mixed-20261109/mixed-policy-fp32-tanh.onnx /data/local/tmp/jetlink-history-160 $remote/shared-boundaries-out $Frames $remote/shared-boundaries.csv; rc=`$?; echo RUNNER_EXIT=`$rc; exit `$rc'"
$cmd | Set-Content -LiteralPath (Join-Path $logs 'command.txt') -Encoding utf8
$text=& $adb -s $Serial shell $cmd 2>&1; $adbExit=$LASTEXITCODE
$text | Set-Content -LiteralPath (Join-Path $logs 'stdout.txt') -Encoding utf8
$adbExit | Set-Content -LiteralPath (Join-Path $logs 'adb-exitcode.txt') -Encoding ascii
(Adb @('shell','su -c ''cat /sys/class/kgsl/kgsl-3d0/temp''')) | Set-Content -LiteralPath (Join-Path $logs 'shared-boundaries.gpu-after.txt') -Encoding ascii
(Adb @('shell','dumpsys battery; cat /proc/uptime')) | Set-Content -LiteralPath (Join-Path $logs 'after.txt') -Encoding utf8
if($adbExit -ne 0 -or ($text -join "`n") -notmatch 'RUNNER_EXIT=0\s*$'){$text | Select-Object -Last 24 | Write-Output; throw 'Mixed runner failed; see stdout.txt'}
Adb @('pull',"$remote/shared-boundaries.csv",(Join-Path $logs 'shared-boundaries.csv')) | Out-Null
Adb @('pull',"$remote/shared-boundaries-out",$raw) | Out-Null
$text | Select-Object -Last 5 | Write-Output
Write-Output "Finished $Frames mixed frames: $logs"
