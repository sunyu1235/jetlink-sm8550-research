param(
    [string]$Serial = '192.168.1.50:5555',
    [string]$RunId = ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)
$ErrorActionPreference = 'Stop'
$adb = 'C:\mp-sdk\platform-tools\adb.exe'
$root = (Get-Location).Path
$report = Join-Path $root 'reports\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
$models = Join-Path $root 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109'
if ($RunId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunId must contain only letters, numbers, underscores or hyphens' }
$localRun = Join-Path $models $RunId
$logs = Join-Path $report $RunId
$remote = "/data/local/tmp/jetlink-partition-tuning-20261109/$RunId"
if ((Test-Path -LiteralPath $localRun) -or (Test-Path -LiteralPath $logs)) { throw 'Refusing to overwrite an existing run' }
function Invoke-Adb {
    param([string[]]$AdbArgs)
    $result = & $adb -s $Serial @AdbArgs 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ADB failed: $($AdbArgs -join ' ')`n$($result -join "`n")" }
    return ($result -join "`n").Trim()
}
function Read-GpuTemp {
    $v = Invoke-Adb @('shell', 'su -c ''cat /sys/class/kgsl/kgsl-3d0/temp''')
    $n = 0
    if (![int]::TryParse($v.Trim(), [ref]$n) -or $n -lt 0) { throw "GPU temperature unavailable: $v" }
    return $n
}
function Wait-Cool {
    $gpu = Read-GpuTemp
    if ($gpu -ge 60000) { Write-Output "Cooling from GPU sensor ${gpu}mC"; Start-Sleep -Seconds 25; $gpu = Read-GpuTemp }
    if ($gpu -ge 60000) { throw "GPU sensor remains ${gpu}mC; stop until the phone cools" }
}
function Run-Probe {
    param([string]$Name, [string]$Model, [string]$InputDir, [int]$Frames, [bool]$Capture, [string]$Soc)
    Wait-Cool
    $gpuBefore = Read-GpuTemp
    $output = "$remote/$Name-out"
    $csv = "$remote/$Name.csv"
    $ctx = if ($Capture) { '' } else { "export JL_CONTEXT_OUTPUT=$remote/$Name-ctx.onnx;" }
    $custom = if ($Capture) { 'export JL_CUSTOM_OP_LIBRARY=/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so;' } else { 'unset JL_CUSTOM_OP_LIBRARY;' }
    $socEnv = if ($Soc) { "export JL_SOC_MODEL=$Soc;" } else { 'unset JL_SOC_MODEL;' }
    $modulo = if ($Capture) { 'unset JL_INPUT_MODULO;' } else { 'export JL_INPUT_MODULO=8;' }
    $dump = 'gelu_42,gelu_53,gelu_60,add_48,add_68,add_84,add_50,add_70,add_86,linear_106,linear_147,linear_176'
    $cmd = "mkdir $output && taskset f8 /system/bin/sh -c 'echo CHILD_STATUS; grep Cpus_allowed /proc/self/status; export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:${libPath}:/vendor/lib64; export ADSP_LIBRARY_PATH=$libPath; unset JL_CONTEXT_OUTPUT JL_VTCM_MB JL_ORT_PROFILE; $custom $socEnv $modulo $ctx export JL_DUMP_TENSORS=$dump; export JL_DUMP_FRAME_LIMIT=8; export JL_FRAME_CSV=$csv; $remote/partition_probe $Model 1 0 none $Frames $InputDir $output varied; rc=`$?; echo RUNNER_EXIT=`$rc; exit `$rc'"
    $cmd | Set-Content -LiteralPath (Join-Path $logs "$Name.command.txt") -Encoding utf8
    (Invoke-Adb @('shell','dumpsys battery; dumpsys thermalservice; cat /proc/meminfo')) | Set-Content -LiteralPath (Join-Path $logs "$Name.before.txt") -Encoding utf8
    $text = & $adb -s $Serial shell $cmd 2>&1
    $adbExit = $LASTEXITCODE
    $text | Set-Content -LiteralPath (Join-Path $logs "$Name.stdout.txt") -Encoding utf8
    $adbExit | Set-Content -LiteralPath (Join-Path $logs "$Name.adb-exitcode.txt") -Encoding ascii
    $runnerOk = ($text -join "`n") -match 'RUNNER_EXIT=0\s*$'
    if ($adbExit -ne 0 -or !$runnerOk) { throw "$Name did not finish successfully; stop without starting another probe" }
    $gpuAfter = Read-GpuTemp
    @{ gpu_before_milli_c=$gpuBefore; gpu_after_milli_c=$gpuAfter; frames=$Frames; model=$Model; soc_model=$Soc; capture=$Capture } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs "$Name.metadata.json") -Encoding utf8
    (Invoke-Adb @('shell','dumpsys battery; dumpsys thermalservice; cat /proc/meminfo')) | Set-Content -LiteralPath (Join-Path $logs "$Name.after.txt") -Encoding utf8
    Invoke-Adb @('pull', $csv, (Join-Path $logs "$Name.csv")) | Out-Null
    Invoke-Adb @('pull', $output, $localRun) | Out-Null
    Write-Output "$Name completed: GPU sensor ${gpuBefore}->${gpuAfter}mC"
    if ($gpuAfter -ge 75000) { throw "Stop after $Name at ${gpuAfter}mC" }
}

$state = Invoke-Adb @('get-state')
if ($state -ne 'device') { throw "Device is not ready: $state" }
Invoke-Adb @('shell','test -f /data/local/tmp/light-speed-gelu-both-ctx_qnn.bin && test -f /data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so && test -f /data/local/tmp/jetlink-history-160/0_new_img.bin') | Out-Null
$apk = Invoke-Adb @('shell','pm path io.zoompilot.jetlink.android')
$baseApk = @($apk -split "`r?`n" | Where-Object { $_ -match 'base.apk$' })[0] -replace '^package:',''
if (!$baseApk -or $baseApk -notmatch '^/data/app/[a-zA-Z0-9_~+=./-]+/base.apk$') { throw "Unexpected app path: $apk" }
$libPath = $baseApk -replace '/base.apk$','/lib/arm64'
Invoke-Adb @('shell',"test -d $libPath && test ! -e $remote && mkdir -p $remote") | Out-Null
New-Item -ItemType Directory -Path $logs,$localRun | Out-Null
(Invoke-Adb @('shell','getprop ro.product.model; getprop ro.boot.bootreason; getprop sys.boot.reason; cat /proc/uptime; cat /proc/sys/kernel/random/boot_id')) | Set-Content -LiteralPath (Join-Path $logs 'device-start.txt') -Encoding utf8
foreach ($file in @('partition_probe','partition-130-40.onnx','accepted-context-capture-130-40.onnx')) {
    $source = Join-Path $models $file
    Invoke-Adb @('push', $source, "$remote/$file") | Out-Null
    $deviceHash = (Invoke-Adb @('shell',"sha256sum $remote/$file")).Split(' ')[0].Trim()
    $localHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($deviceHash -ne $localHash) { throw "Hash mismatch for $file" }
}
Invoke-Adb @('shell',"chmod 755 $remote/partition_probe") | Out-Null
Run-Probe -Name 'capture8' -Model "$remote/accepted-context-capture-130-40.onnx" -InputDir '/data/local/tmp/jetlink-history-160' -Frames 8 -Capture $true -Soc ''
& python (Join-Path $report 'analyze_partition.py') --capture-only --capture-dir (Join-Path $localRun 'capture8-out') --frames 8 --output (Join-Path $logs 'capture-check.json')
if ($LASTEXITCODE -ne 0) { throw 'Capture validation failed' }
$captureCheck = Get-Content -LiteralPath (Join-Path $logs 'capture-check.json') -Raw | ConvertFrom-Json
if (!$captureCheck.all_bits_match -or !$captureCheck.all_finite -or !$captureCheck.all_files_present) { throw 'Capture differs from the accepted precise baseline or is incomplete' }
foreach ($run in @(@{name='a0-default';soc=''},@{name='b1-soc43';soc='43'},@{name='a2-default';soc=''})) {
    Run-Probe -Name $run.name -Model "$remote/partition-130-40.onnx" -InputDir "$remote/capture8-out" -Frames 160 -Capture $false -Soc $run.soc
    & python (Join-Path $report 'analyze_partition.py') --capture-dir (Join-Path $localRun 'capture8-out') --run-dir (Join-Path $localRun ($run.name+'-out')) --frames 8 --input-modulo 8 --output (Join-Path $logs ($run.name+'-check.json'))
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed for $($run.name)" }
    $check = Get-Content -LiteralPath (Join-Path $logs ($run.name+'-check.json')) -Raw | ConvertFrom-Json
    if (!$check.all_files_present -or !$check.all_finite -or !$check.allclose) { throw "Subgraph output check failed for $($run.name); keep results and stop" }
}
Write-Output "A-B-A complete. Evidence: $logs"
