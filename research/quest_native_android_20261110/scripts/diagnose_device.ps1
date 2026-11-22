param([switch]$SignedPd, [switch]$FullPreview)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $root 'tools/android-sdk/platform-tools/adb.exe'
& $adb get-state
if ($LASTEXITCODE -ne 0) { throw 'Quest ADB device is unavailable.' }
& $adb shell am force-stop ai.mycartgs.quest
$launch = @('shell','am','start','-n','ai.mycartgs.quest/.QuestActivity',
            '--ez','mycartgs.validation_load','true')
if ($SignedPd) { $launch += @('--ez','mycartgs.validation_signed_pd','true') }
if ($FullPreview) { $launch += @('--ez','mycartgs.validation_full','true') }
& $adb @launch
if ($LASTEXITCODE -ne 0) { throw 'Quest Activity launch failed.' }
Write-Output '保持佩戴头显以进入 OpenXR；Ctrl+C 结束日志。此命令不修改系统权限，不启用 MLP 回退。'
& $adb logcat -v threadtime 'MyCartGS:I' 'MyCartGS-QNN:I' 'dsp-client:E' '*:S'
