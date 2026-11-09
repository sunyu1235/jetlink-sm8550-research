$ErrorActionPreference='Stop'
$adb='C:\adb\adb.exe';$serial='192.168.1.49:5555'
$report=Join-Path (Get-Location) 'reports\jetlink-gelu-optimization-20261101\threads'
$rawroot=Join-Path (Get-Location) 'outputs\jetlink-gelu-optimization-20261101\threads'
$remoteBase='/data/local/tmp/jetlink-gelu-optimization-20261101/threads'
$libPath='/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64'
$runner="$remoteBase/history_probe_threads"
$model='/data/local/tmp/light-speed-gelu-both-ctx.onnx'
$custom='/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so'
$input='/data/local/tmp/jetlink-history-160'
$runs=@(
 @{Name='a0-threads4';Threads=4},
 @{Name='b1-threads1';Threads=1},
 @{Name='a2-threads4';Threads=4}
)
foreach($r in $runs){
 $name=$r.Name;$threads=$r.Threads
 $localRaw=Join-Path $rawroot $name;New-Item -ItemType Directory -Force -Path $localRaw|Out-Null
 $remoteOut="$remoteBase/$name-out";$remoteCsv="$remoteBase/$name.csv"
 $pre=& $adb -s $serial shell "test ! -e $remoteOut -a ! -e $remoteCsv"
 if($LASTEXITCODE -ne 0){throw "refusing to overwrite existing remote artifacts for $name"}
 $hostStart=(Get-Date).ToUniversalTime().ToString('o');$hostStart|Set-Content -LiteralPath (Join-Path $report "$name.host-start.txt") -Encoding ascii
 $gpuBefore=& $adb -s $serial shell su -c 'cat /sys/class/kgsl/kgsl-3d0/temp'
 $gpuBefore|Set-Content -LiteralPath (Join-Path $report "$name.gpu-before.txt") -Encoding ascii
 $gpuN=-1;[int]::TryParse(($gpuBefore -join '').Trim(),[ref]$gpuN)|Out-Null
 if($gpuN -ge 60000){"STOP before ${name}: GPU sensor ${gpuN}mC >= 60000mC"|Set-Content -LiteralPath (Join-Path $report 'thermal-stop.txt') -Encoding utf8;throw "thermal stop before ${name} at ${gpuN}mC"}
 $batteryBefore=& $adb -s $serial shell dumpsys battery
 $batteryBefore|Set-Content -LiteralPath (Join-Path $report "$name.battery-before.txt") -Encoding utf8
 $remoteCmd="mkdir -p $remoteOut; taskset f8 /system/bin/sh -c 'echo CHILD_STATUS; grep Cpus_allowed /proc/self/status; export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:${libPath}:/vendor/lib64; export ADSP_LIBRARY_PATH=$libPath; export JL_CUSTOM_OP_LIBRARY=$custom; export JL_CPU_THREADS=$threads; export JL_FRAME_CSV=$remoteCsv; $runner $model 1 0 none 160 $input $remoteOut varied; rc=`$?; echo RUNNER_EXIT=`$rc; exit `$rc'"
 $remoteCmd|Set-Content -LiteralPath (Join-Path $report "$name.command.txt") -Encoding utf8
 $out=& $adb -s $serial shell $remoteCmd 2>&1;$adbExit=$LASTEXITCODE
 $out|Set-Content -LiteralPath (Join-Path $report "$name.stdout.txt") -Encoding utf8
 $adbExit|Set-Content -LiteralPath (Join-Path $report "$name.adb-exitcode.txt") -Encoding ascii
 (Get-Date).ToUniversalTime().ToString('o')|Set-Content -LiteralPath (Join-Path $report "$name.host-end.txt") -Encoding ascii
 $gpuAfter=& $adb -s $serial shell su -c 'cat /sys/class/kgsl/kgsl-3d0/temp'
 $gpuAfter|Set-Content -LiteralPath (Join-Path $report "$name.gpu-after.txt") -Encoding ascii
 $batteryAfter=& $adb -s $serial shell dumpsys battery
 $batteryAfter|Set-Content -LiteralPath (Join-Path $report "$name.battery-after.txt") -Encoding utf8
 & $adb -s $serial pull $remoteCsv (Join-Path $report "$name.csv") | Out-Null
 & $adb -s $serial pull $remoteOut $localRaw | Out-Null
 $combined=$out -join "`n";$remoteExit=[regex]::Match($combined,'RUNNER_EXIT=(\d+)').Groups[1].Value
 "DONE $name threads=$threads adb_exit=$adbExit runner_exit=$remoteExit first_csv=$((Get-Content -LiteralPath (Join-Path $report "$name.csv") -TotalCount 2|Select-Object -Last 1)) gpu=$($gpuBefore -join '')->$($gpuAfter -join '')"
 if($adbExit -ne 0 -or $remoteExit -ne '0'){throw "runner failed for $name; stopping A-B-A"}
 $gpuAfterN=-1;[int]::TryParse(($gpuAfter -join '').Trim(),[ref]$gpuAfterN)|Out-Null
 if($gpuAfterN -ge 60000 -and $name -ne 'a2-threads4'){"STOP after ${name}: GPU sensor ${gpuAfterN}mC >= 60000mC"|Set-Content -LiteralPath (Join-Path $report 'thermal-stop.txt') -Encoding utf8;throw "thermal stop after ${name} at ${gpuAfterN}mC"}
 if($name -ne 'a2-threads4'){Start-Sleep -Seconds 25}
}

