$ErrorActionPreference='Stop'
$adb='Q:/tools/android-sdk/platform-tools/adb.exe'
& $adb -s 2G97C5ZJ2G00PX install -r T:/delivery/mycartgs-splatkit-baseline.apk
if($LASTEXITCODE -ne 0){throw 'APK install failed'}
& $adb -s 2G97C5ZJ2G00PX shell am start -n ai.mycartgs.quest.splatkit/ai.mycartgs.quest.QuestActivity
if($LASTEXITCODE -ne 0){throw 'APK launch failed'}
Write-Output 'SplatKit 独立包已启动。选择 CPU INFERENCE TEST 测CPU，GPU POINT COUNT TEST测渲染；原应用保留。'
