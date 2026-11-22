$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $root 'tools/android-sdk/platform-tools/adb.exe'
$apk = Join-Path $root 'delivery/mycartgs-quest-xnnpack-cpu-release.apk'
if (!(Test-Path $apk)) { throw 'Build the Release APK first.' }
& $adb get-state
if ($LASTEXITCODE -ne 0) { throw 'Quest ADB is unavailable.' }
& $adb install -r $apk
if ($LASTEXITCODE -ne 0) { throw 'APK installation failed.' }
& $adb shell am force-stop ai.mycartgs.quest
& $adb shell am start -n ai.mycartgs.quest/.QuestActivity
if ($LASTEXITCODE -ne 0) { throw 'Activity launch failed.' }
Write-Output '0.4.1 已启动；主界面 VULKAN POINT COUNT TEST 可测试点数，正常实景选择 RUN CPU LOD。'
