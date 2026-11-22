$ErrorActionPreference='Stop'
$cmake='Q:/tools/android-sdk/cmake/3.22.1/bin/cmake.exe'
& $cmake -S T:/tests -B T:/build-cpu-sanitizer -G Ninja `
    -DCMAKE_MAKE_PROGRAM=Q:/tools/android-sdk/cmake/3.22.1/bin/ninja.exe `
    -DCMAKE_TOOLCHAIN_FILE=T:/tools/android-ndk-r27b/build/cmake/android.toolchain.cmake `
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release -DMYCARTGS_CPU_SANITIZE=ON
if($LASTEXITCODE -ne 0){throw 'CPU sanitizer configure failed'}
& $cmake --build T:/build-cpu-sanitizer --target cpu_optimization_probe -j 6
if($LASTEXITCODE -ne 0){throw 'CPU sanitizer build failed'}
