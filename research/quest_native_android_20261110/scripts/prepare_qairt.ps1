$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$sdk=Join-Path $root 'tools/qairt-sdk/qairt/2.40.0.251030'
if (!(Test-Path (Join-Path $sdk 'include/QNN/QnnInterface.h'))) { throw 'QAIRT 2.40 SDK headers missing' }
$target=Join-Path $root 'app/src/main/jniLibs/arm64-v8a'
New-Item -ItemType Directory -Force $target | Out-Null
foreach ($name in @('libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV69Stub.so','libQnnSystem.so')) {
    Copy-Item -LiteralPath (Join-Path $sdk "lib/aarch64-android/$name") -Destination $target -Force
}
foreach ($name in @('libQnnHtpV69Skel.so','libQnnHtpV69.so')) {
    Copy-Item -LiteralPath (Join-Path $sdk "lib/hexagon-v69/unsigned/$name") -Destination $target -Force
}
