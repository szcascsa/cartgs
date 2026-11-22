$ErrorActionPreference = 'Stop'
$studyRoot = Split-Path -Parent $PSScriptRoot
$env:JAVA_HOME = 'Q:/tools/jdk-17.0.21+9'
$env:ANDROID_HOME = Join-Path $studyRoot 'tools/android-sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:GRADLE_USER_HOME = Join-Path $studyRoot '.gradle-user'
& (Join-Path $studyRoot 'tools/gradle-8.13/bin/gradle.bat') --project-dir (Join-Path $studyRoot 'source/SplatSample') `
    --no-daemon --stacktrace :app:assembleRelease :app:testReleaseUnitTest
if ($LASTEXITCODE -ne 0) { throw 'Official SplatSample build/tests failed.' }
$apk = Join-Path $studyRoot 'source/SplatSample/app/build/outputs/apk/release/app-release.apk'
$target = Join-Path $studyRoot 'delivery/mycartgs-meta-official-splat-bench.apk'
Copy-Item -LiteralPath $apk -Destination $target -Force
Write-Output "APK=$target"
Get-FileHash -Algorithm SHA256 -LiteralPath $target
