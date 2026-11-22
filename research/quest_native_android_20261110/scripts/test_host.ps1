$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root 'build-host/native_core_test.exe'
New-Item -ItemType Directory -Force (Split-Path -Parent $output) | Out-Null
& g++ -std=c++17 -O2 -Wall -Wextra -Werror `
    -I (Join-Path $root 'app/src/main/cpp/mycartgs') `
    (Join-Path $root 'tests/native_core_test.cpp') `
    (Join-Path $root 'app/src/main/cpp/mycartgs/decode.cpp') `
    (Join-Path $root 'app/src/main/cpp/mycartgs/sha256.cpp') `
    (Join-Path $root 'app/src/main/cpp/mycartgs/residual48.cpp') `
    (Join-Path $root 'app/src/main/cpp/mycartgs/strict_topk.cpp') `
    -o $output
if ($LASTEXITCODE -ne 0) { throw 'Host C++ compilation failed.' }
& $output
if ($LASTEXITCODE -ne 0) { throw 'Host C++ test failed.' }
& g++ -std=c++17 -O2 -Wall -Wextra -Werror `
    -I (Join-Path $root 'app/src/main/cpp/mycartgs') `
    (Join-Path $root 'tests/scene_menu_test.cpp') `
    (Join-Path $root 'app/src/main/cpp/mycartgs/scene_menu.cpp') `
    -o (Join-Path $root 'build-host/scene_menu_test.exe')
if ($LASTEXITCODE -ne 0) { throw 'Scene menu test compilation failed.' }
& (Join-Path $root 'build-host/scene_menu_test.exe') (Join-Path $root 'build-host')
if ($LASTEXITCODE -ne 0) { throw 'Scene menu test failed.' }
& g++ -std=c++17 -O2 -Wall -Wextra -Werror `
    -I (Join-Path $root 'app/src/main/cpp/mycartgs') `
    (Join-Path $root 'tests/point_benchmark_test.cpp') `
    -o (Join-Path $root 'build-host/point_benchmark_test.exe')
if ($LASTEXITCODE -ne 0) { throw 'Point benchmark test compilation failed.' }
& (Join-Path $root 'build-host/point_benchmark_test.exe')
if ($LASTEXITCODE -ne 0) { throw 'Point benchmark test failed.' }
