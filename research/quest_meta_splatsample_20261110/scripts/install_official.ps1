$ErrorActionPreference = 'Stop'
$studyRoot = Split-Path -Parent $PSScriptRoot
$adb = 'Q:/tools/android-sdk/platform-tools/adb.exe'
$apk = Join-Path $studyRoot 'delivery/mycartgs-meta-official-splat-bench.apk'
& $adb -s 2G97C5ZJ2G00PX install -r $apk
if ($LASTEXITCODE -ne 0) { throw 'Official benchmark installation failed.' }
& $adb -s 2G97C5ZJ2G00PX shell am force-stop ai.mycartgs.quest.official
& $adb -s 2G97C5ZJ2G00PX shell am start -n ai.mycartgs.quest.official/com.meta.spatial.samples.splatsample.SplatSampleActivity
if ($LASTEXITCODE -ne 0) { throw 'Official benchmark launch failed.' }
Write-Output 'Official Splat 测试版已启动；原 ai.mycartgs.quest 保留。戴上头显后确认场景，再选择两轮自动测速。'
