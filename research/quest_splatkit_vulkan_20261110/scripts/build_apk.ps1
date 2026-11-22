$ErrorActionPreference='Stop'
$env:JAVA_HOME='Q:/tools/jdk-17.0.21+9'
$env:ANDROID_HOME='Q:/tools/android-sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:GRADLE_USER_HOME='T:/.gradle-user'
& 'Q:/tools/gradle-8.10.2/bin/gradle.bat' --project-dir T:/host --no-daemon --stacktrace assembleRelease
if($LASTEXITCODE -ne 0){throw 'SplatKit APK build failed'}
Copy-Item -LiteralPath 'T:/host/app/build/outputs/apk/release/app-release.apk' -Destination 'T:/delivery/mycartgs-splatkit-baseline.apk'
Get-FileHash -LiteralPath 'T:/delivery/mycartgs-splatkit-baseline.apk' -Algorithm SHA256
