param(
    [string]$JdkHome = '',
    [string]$AndroidSdkRoot = 'C:\mp-sdk'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
$Source = Join-Path $PSScriptRoot 'Native.java'
$OutputDir = Join-Path $ProjectRoot 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109\gpu-converter'
$OutputJar = Join-Path $OutputDir 'native-console.jar'
$Manifest = Join-Path $PSScriptRoot 'build-manifest.json'

if (-not $JdkHome) {
    $Candidates = @(
        $env:JAVA_HOME,
        (Join-Path $ProjectRoot 'tools\jdk-17'),
        'C:\Program Files\Android\Android Studio\jbr',
        'C:\Program Files\Android\Android Studio\jre'
    )
    $PathJavac = Get-Command javac.exe -ErrorAction SilentlyContinue
    if ($PathJavac) {
        $Candidates += Split-Path (Split-Path $PathJavac.Source -Parent) -Parent
    }
    $JdkHome = $Candidates | Where-Object {
        $_ -and (Test-Path -LiteralPath (Join-Path $_ 'bin\javac.exe'))
    } | Select-Object -First 1
}
if (-not $JdkHome) { throw 'No JDK found; pass -JdkHome.' }
$JdkHome = [IO.Path]::GetFullPath($JdkHome)
$Javac = Join-Path $JdkHome 'bin\javac.exe'
if (-not (Test-Path -LiteralPath $Javac)) { throw "javac missing: $Javac" }
$Java = Join-Path $JdkHome 'bin\java.exe'
$JavacVersion = ((& $Javac -version 2>&1) | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to query javac version' }

$D8 = Get-ChildItem -LiteralPath (Join-Path $AndroidSdkRoot 'build-tools') -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'd8.bat') } |
    Sort-Object { [version](($_.Name -split '-')[0]) } -Descending |
    Select-Object -First 1
if (-not $D8) { throw "No d8.bat found under $AndroidSdkRoot" }
$D8Path = Join-Path $D8.FullName 'd8.bat'
$D8Jar = $null
if ((Get-Content -LiteralPath $D8Path -Raw) -match '-Djava\.ext\.dirs=' -and $JavacVersion -notmatch '^javac 1\.[0-8][\.]') {
    # This SDK wrapper predates Java 9; use its unchanged D8 entry point.
    $D8Jar = @('d8.jar', 'lib\d8.jar', '..\framework\d8.jar') |
        ForEach-Object { Join-Path $D8.FullName $_ } |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $D8Jar) { throw "No d8.jar found beside $D8Path" }
}
$Platform = Get-ChildItem -LiteralPath (Join-Path $AndroidSdkRoot 'platforms') -Directory |
    Where-Object { $_.Name -match '^android-\d+$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'android.jar')) } |
    Sort-Object { [int]($_.Name -replace '^android-', '') } -Descending |
    Select-Object -First 1
if (-not $Platform) { throw "No android.jar found under $AndroidSdkRoot" }
$AndroidJar = Join-Path $Platform.FullName 'android.jar'

$BuildDir = Join-Path ([IO.Path]::GetTempPath()) ('jetlink-native-console-' + [guid]::NewGuid().ToString('N'))
$ClassesDir = Join-Path $BuildDir 'classes'
$TempZip = Join-Path $BuildDir 'native-console.zip'
$PreviousJavaHome = $env:JAVA_HOME
try {
    New-Item -ItemType Directory -Path $ClassesDir -Force | Out-Null
    $env:JAVA_HOME = $JdkHome
    & $Javac -encoding UTF-8 -source 8 -target 8 -Xlint:-options -classpath $AndroidJar -d $ClassesDir $Source
    if ($LASTEXITCODE -ne 0) { throw "javac failed: $LASTEXITCODE" }
    $ClassFiles = @(Get-ChildItem -LiteralPath $ClassesDir -Filter '*.class' -Recurse | ForEach-Object { $_.FullName })
    if ($D8Jar) {
        & $Java -Xmx1g -cp $D8Jar com.android.tools.r8.D8 --min-api 26 --lib $AndroidJar --output $TempZip @ClassFiles
    } else {
        & $D8Path --min-api 26 --lib $AndroidJar --output $TempZip @ClassFiles
    }
    if ($LASTEXITCODE -ne 0) { throw "d8 failed: $LASTEXITCODE" }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [IO.Compression.ZipFile]::OpenRead($TempZip)
    try {
        $Entries = @($Archive.Entries | ForEach-Object { $_.FullName })
        if ($Entries -notcontains 'classes.dex') { throw 'd8 archive has no classes.dex' }
    } finally {
        $Archive.Dispose()
    }
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Copy-Item -LiteralPath $TempZip -Destination $OutputJar -Force

    $PinnedRoot = Join-Path $ProjectRoot 'outputs\jetlink-gelu-optimization-20261101\partition-tuning-20261109\upstream-pinned-source'
    $Bridge = Join-Path $PinnedRoot 'JetlinkKit\Sources\JetlinkAndroid\Bridge.swift'
    $KotlinNative = Join-Path $PinnedRoot 'android\app\src\main\java\io\zoompilot\jetlink\server\Native.kt'
    $Data = [ordered]@{
        built_at_utc = [DateTime]::UtcNow.ToString('o')
        source = $Source
        source_sha256 = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
        bridge_source = $Bridge
        bridge_sha256 = (Get-FileHash -LiteralPath $Bridge -Algorithm SHA256).Hash.ToLowerInvariant()
        kotlin_source = $KotlinNative
        kotlin_sha256 = (Get-FileHash -LiteralPath $KotlinNative -Algorithm SHA256).Hash.ToLowerInvariant()
        javac = $Javac
        javac_version = $JavacVersion
        android_jar = $AndroidJar
        d8 = $D8Path
        d8_jar_fallback = $D8Jar
        java_source_target = 8
        min_api = 26
        output = $OutputJar
        output_sha256 = (Get-FileHash -LiteralPath $OutputJar -Algorithm SHA256).Hash.ToLowerInvariant()
        output_bytes = (Get-Item -LiteralPath $OutputJar).Length
        archive_entries = $Entries
        entry_class = 'io.zoompilot.jetlink.server.Native'
        jni_methods = @('static String start(String)', 'static void stop()', 'static String command(String)', 'static String snapshot(long,int)', 'static String logs(long)')
        config = [ordered]@{cache = '<cacheDir>'; backend = 'litert'; device = 'gpu'; chip = 'SM8550'; native_library_dir = '<nativeLibDir>'; preload = $false; listen = $false; port = 0; usb = $false; keep_alive = $false; keep_cpu_warm = $false}
        stdin_commands = @('<control JSON>', '@start-snapshot', '@logs', '@quit')
        app_process = 'CLASSPATH=/data/local/tmp/native-console.jar LD_LIBRARY_PATH="$LIBDIR" app_process /system/bin io.zoompilot.jetlink.server.Native "$LIBDIR" "$CACHEDIR"'
        device_connected = $false
    }
    $Data | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Manifest -Encoding UTF8
    Write-Output $OutputJar
    Write-Output ('SHA256=' + $Data.output_sha256)
} finally {
    $env:JAVA_HOME = $PreviousJavaHome
    $TempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    $ResolvedBuildDir = [IO.Path]::GetFullPath($BuildDir)
    if (-not $ResolvedBuildDir.StartsWith($TempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup outside temp root: $ResolvedBuildDir"
    }
    if (Test-Path -LiteralPath $ResolvedBuildDir) {
        Remove-Item -LiteralPath $ResolvedBuildDir -Recurse -Force
    }
}
