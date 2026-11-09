$ErrorActionPreference='Stop'
$adb='C:\adb\adb.exe';$serial='192.168.1.49:5555'
$runner='/data/local/tmp/jetlink-gelu-optimization-20261101/threads/history_probe_threads'
$report=Join-Path (Get-Location) 'reports\jetlink-gelu-optimization-20261101\threads'
foreach($v in @('0','9','bad')){
  $cmd="JL_CPU_THREADS=$v JL_CUSTOM_OP_LIBRARY=/not-loaded $runner /not-loaded/model.onnx; rc=`$?; echo PREFLIGHT_EXIT=`$rc"
  $out=& $adb -s $serial shell $cmd 2>&1
  $code=$LASTEXITCODE
  $out | Set-Content -LiteralPath (Join-Path $report "range-reject-$v.log") -Encoding utf8
  if($code -ne 0 -or (($out -join "`n") -notmatch 'PREFLIGHT_EXIT=2') -or (($out -join "`n") -notmatch 'must be an integer in \[1,8\]')){throw "range preflight failed for $v; adb=$code; output=$out"}
  "rejected $v before ORT/session, adb_exit=$code"
}
