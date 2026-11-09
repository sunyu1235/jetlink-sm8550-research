param([string]$Serial='ZY22GVVTVZ', [string]$RunId=('shared-'+(Get-Date -Format 'yyyyMMdd-HHmmss')), [int]$Frames=16, [switch]$CandidateOnly, [string]$ModelFile='whole-context-shared-gelu-boundaries.onnx', [ValidateSet('none','basic')][string]$Profile='none', [string]$DumpTensors='')
$ErrorActionPreference='Stop'
$adb='C:\mp-sdk\platform-tools\adb.exe'
$root=(Get-Location).Path
$report=Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$models=Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$logs=Join-Path $report $RunId
$raw=Join-Path $models $RunId
$remote="/data/local/tmp/jetlink-shared-boundaries-20261109/$RunId"
$remoteModel="/data/local/tmp/$RunId.onnx"
if($RunId -notmatch '^[a-zA-Z0-9_-]+$' -or $Frames -lt 1 -or $Frames -gt 160){throw 'Invalid run identifier/frame count'}
if($ModelFile -notmatch '^[a-zA-Z0-9_-]+\.onnx$'){throw 'Invalid model filename'}
if($DumpTensors -notmatch '^[a-zA-Z0-9_,]*$'){throw 'Invalid tensor list'}
if((Test-Path -LiteralPath $logs) -or (Test-Path -LiteralPath $raw)){throw 'Refusing to overwrite an existing run'}
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
Adb @('shell',"test ! -e $remote && test ! -e $remoteModel && mkdir -p $remote") | Out-Null
New-Item -ItemType Directory -Path $logs,$raw | Out-Null
foreach($file in @('partition_probe',$ModelFile)){
 $p=Join-Path $models $file
 $target=if($file -eq 'partition_probe'){"$remote/$file"}else{$remoteModel}
 Adb @('push',$p,$target) | Out-Null
 $hash=(Adb @('shell',"sha256sum $target")).Split(' ')[0]
 if($hash -ne (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()){throw "Hash mismatch: $file"}
}
Adb @('shell',"chmod 755 $remote/partition_probe") | Out-Null
$runs=@(@{name='shared-boundaries';model=$remoteModel})
if(!$CandidateOnly){$runs=@(@{name='baseline';model='/data/local/tmp/light-speed-gelu-both-ctx.onnx'})+$runs}
foreach($run in $runs){
 $name=$run.name
 $temp=Adb @('shell','su -c ''cat /sys/class/kgsl/kgsl-3d0/temp''')
 $tempN=0
 if(![int]::TryParse($temp,[ref]$tempN) -or $tempN -ge 60000){throw "Temperature unavailable or >=60C: $temp"}
 $temp | Set-Content -LiteralPath (Join-Path $logs "$name.gpu-before.txt") -Encoding ascii
 (Adb @('shell','dumpsys thermalservice; dumpsys battery; cat /proc/meminfo | head -n 6; cat /proc/uptime')) | Set-Content -LiteralPath (Join-Path $logs "$name.before.txt") -Encoding utf8
 $out="$remote/$name-out"; $csv="$remote/$name.csv"
 $dumpEnv=if($DumpTensors){"export JL_DUMP_TENSORS=$DumpTensors; export JL_DUMP_FRAME_LIMIT=8;"}else{''}
 $cmd="mkdir $out && taskset f8 /system/bin/sh -c 'grep Cpus_allowed /proc/self/status; export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:${lib}:/vendor/lib64; export ADSP_LIBRARY_PATH=$lib; export JL_CUSTOM_OP_LIBRARY=/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so; unset JL_CONTEXT_OUTPUT JL_SOC_MODEL JL_VTCM_MB JL_ORT_PROFILE JL_DUMP_TENSORS JL_DUMP_FRAME_LIMIT JL_INPUT_MODULO; $dumpEnv export JL_QNN_PROFILE_PATH=$remote/$name-profile.csv; export JL_FRAME_CSV=$csv; $remote/partition_probe $($run.model) 1 0 $Profile $Frames /data/local/tmp/jetlink-history-160 $out varied; rc=`$?; echo RUNNER_EXIT=`$rc; exit `$rc'"
 $cmd | Set-Content -LiteralPath (Join-Path $logs "$name.command.txt") -Encoding utf8
 $text=& $adb -s $Serial shell $cmd 2>&1; $adbExit=$LASTEXITCODE
 $text | Set-Content -LiteralPath (Join-Path $logs "$name.stdout.txt") -Encoding utf8
 $adbExit | Set-Content -LiteralPath (Join-Path $logs "$name.adb-exitcode.txt") -Encoding ascii
 if($adbExit -ne 0 -or ($text -join "`n") -notmatch 'RUNNER_EXIT=0\s*$'){throw "$name failed; stop"}
 (Adb @('shell','su -c ''cat /sys/class/kgsl/kgsl-3d0/temp''')) | Set-Content -LiteralPath (Join-Path $logs "$name.gpu-after.txt") -Encoding ascii
 (Adb @('shell','dumpsys thermalservice; dumpsys battery; cat /proc/meminfo | head -n 6; cat /proc/uptime')) | Set-Content -LiteralPath (Join-Path $logs "$name.after.txt") -Encoding utf8
 Adb @('pull',$csv,(Join-Path $logs "$name.csv")) | Out-Null
 Adb @('pull',$out,$raw) | Out-Null
 if($Profile -ne 'none'){Adb @('pull',"$remote/$name-profile.csv",(Join-Path $logs "$name-profile.csv")) | Out-Null}
 $text | Select-Object -Last 3 | Write-Output
}
Write-Output "Finished $Frames frames per version: $logs"
