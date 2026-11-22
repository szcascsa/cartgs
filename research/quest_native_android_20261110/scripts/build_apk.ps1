param([ValidateSet('Debug','Release')][string]$Configuration = 'Release')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$sdk = Join-Path $root 'tools/android-sdk'
$jdk = Join-Path $root 'tools/jdk-17.0.21+9'
$gradle = Join-Path $root 'tools/gradle-8.10.2/bin/gradle.bat'
if (!(Test-Path $gradle) -or !(Test-Path (Join-Path $sdk 'ndk/23.2.8568313'))) {
    throw 'Isolated Android toolchain is incomplete.'
}
$env:JAVA_HOME = $jdk
$env:ANDROID_HOME = $sdk
$env:ANDROID_SDK_ROOT = $sdk
$env:GRADLE_USER_HOME = Join-Path $root '.gradle-user'
& (Join-Path $root 'scripts/build_shaders.ps1')
& python (Join-Path $root 'tools/fetch_xnnpack.py')
if ($LASTEXITCODE -ne 0) { throw 'XNNPACK dependency verification failed.' }
$licenses = Join-Path $root 'app/src/main/assets/licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
foreach ($dependency in @('XNNPACK','cpuinfo','pthreadpool','FXdiv')) {
    Copy-Item (Join-Path $root "third_party/$dependency/LICENSE") (Join-Path $licenses "$dependency.txt") -Force
}
$sdkLine = 'sdk.dir=' + ($sdk -replace '\\','/')
[System.IO.File]::WriteAllText((Join-Path $root 'local.properties'), $sdkLine + [Environment]::NewLine)
$task = if ($Configuration -eq 'Release') { 'assembleRelease' } else { 'assembleDebug' }
& $gradle --project-dir $root --no-daemon --stacktrace $task
if ($LASTEXITCODE -ne 0) { throw "Gradle failed with exit code $LASTEXITCODE" }
$variant = $Configuration.ToLowerInvariant()
$source = Join-Path $root "app/build/outputs/apk/$variant/app-$variant.apk"
if (!(Test-Path $source)) { throw "APK was not produced: $source" }
$delivery = Join-Path $root 'delivery'
New-Item -ItemType Directory -Force $delivery | Out-Null
$target = Join-Path $delivery "mycartgs-quest-xnnpack-cpu-$variant.apk"
Copy-Item -Force $source $target
$hash = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLowerInvariant()
Write-Output "APK=$target"
Write-Output "SHA256=$hash"
