$ErrorActionPreference = 'Stop'
$adbPath = 'C:\adb\adb.exe'
$serial = '192.168.1.49:5555'
$reportRoot = Join-Path (Get-Location) 'reports\jetlink-gelu-optimization-20261101\affinity'
$rawRoot = Join-Path (Get-Location) 'outputs\jetlink-gelu-optimization-20261101\affinity'
$remoteBase = '/data/local/tmp/jetlink-gelu-optimization-20261101/affinity'
$libPath = '/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64'
$model = '/data/local/tmp/light-speed-gelu-both-ctx.onnx'
$runner = '/data/local/tmp/jetlink-gelu-repair-20261101/history_probe'
$custom = '/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so'
$inputDir = '/data/local/tmp/jetlink-history-160'
$runs = @(
  @{ Name = 'a0-all-ff'; Mask = 'ff' },
  @{ Name = 'b1-nonlittle-f8'; Mask = 'f8' },
  @{ Name = 'a2-all-ff'; Mask = 'ff' }
)
foreach ($run in $runs) {
  $name = $run.Name
  $mask = $run.Mask
  $localRaw = Join-Path $rawRoot $name
  New-Item -ItemType Directory -Force -Path $localRaw | Out-Null
  $remoteOut = "$remoteBase/$name-out"
  $remoteCsv = "$remoteBase/$name.csv"
  $commandFile = Join-Path $reportRoot "$name.command.txt"
  $startFile = Join-Path $reportRoot "$name.host-start.txt"
  $stdoutFile = Join-Path $reportRoot "$name.stdout.txt"
  $exitFile = Join-Path $reportRoot "$name.adb-exitcode.txt"
  $gpuBeforeFile = Join-Path $reportRoot "$name.gpu-before.txt"
  $gpuAfterFile = Join-Path $reportRoot "$name.gpu-after.txt"
  $batteryBeforeFile = Join-Path $reportRoot "$name.battery-before.txt"
  $batteryAfterFile = Join-Path $reportRoot "$name.battery-after.txt"
  $csvLocal = Join-Path $reportRoot "$name.csv"
  $hostEndFile = Join-Path $reportRoot "$name.host-end.txt"
  (Get-Date).ToUniversalTime().ToString('o') | Set-Content -LiteralPath $startFile -Encoding ascii
  $gpuBefore = & $adbPath -s $serial shell su -c 'cat /sys/class/kgsl/kgsl-3d0/temp; cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq'
  $gpuBefore | Set-Content -LiteralPath $gpuBeforeFile -Encoding ascii
  $batteryBefore = & $adbPath -s $serial shell dumpsys battery
  $batteryBefore | Set-Content -LiteralPath $batteryBeforeFile -Encoding ascii
  $remoteCmd = "mkdir -p $remoteOut; rm -f $remoteCsv; taskset $mask /system/bin/sh -c 'echo CHILD_STATUS; grep Cpus_allowed /proc/self/status; export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:${libPath}:/vendor/lib64; export ADSP_LIBRARY_PATH=$libPath; export JL_CUSTOM_OP_LIBRARY=$custom; export JL_FRAME_CSV=$remoteCsv; $runner $model 1 0 none 160 $inputDir $remoteOut varied; s=`$?; echo RUNNER_EXIT=`$s; exit `$s'"
  $remoteCmd | Set-Content -LiteralPath $commandFile -Encoding utf8
  $result = & $adbPath -s $serial shell $remoteCmd 2>&1
  $adbExit = $LASTEXITCODE
  $result | Set-Content -LiteralPath $stdoutFile -Encoding utf8
  $adbExit | Set-Content -LiteralPath $exitFile -Encoding ascii
  (Get-Date).ToUniversalTime().ToString('o') | Set-Content -LiteralPath $hostEndFile -Encoding ascii
  $gpuAfter = & $adbPath -s $serial shell su -c 'cat /sys/class/kgsl/kgsl-3d0/temp; cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq'
  $gpuAfter | Set-Content -LiteralPath $gpuAfterFile -Encoding ascii
  $batteryAfter = & $adbPath -s $serial shell dumpsys battery
  $batteryAfter | Set-Content -LiteralPath $batteryAfterFile -Encoding ascii
  & $adbPath -s $serial pull $remoteCsv $csvLocal | Out-Null
  & $adbPath -s $serial pull $remoteOut $localRaw | Out-Null
  "DONE $name adb_exit=$adbExit remote_exit=$((Select-String -LiteralPath $stdoutFile -Pattern 'RUNNER_EXIT=' | Select-Object -Last 1).Line) first_row=$((Get-Content -LiteralPath $csvLocal -TotalCount 2 | Select-Object -Last 1)) gpu_before=$($gpuBefore -join '/') gpu_after=$($gpuAfter -join '/')"
  if ($name -ne 'a2-all-ff') { Start-Sleep -Seconds 20 }
}

