$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$glslc = Join-Path $root 'tools/android-sdk/ndk/23.2.8568313/shader-tools/windows-x86_64/glslc.exe'
$spirvVal = Join-Path $root 'tools/android-sdk/ndk/23.2.8568313/shader-tools/windows-x86_64/spirv-val.exe'
$output = Join-Path $root 'app/src/main/assets/shaders'
if (!(Test-Path $glslc)) { throw "glslc is missing: $glslc" }
New-Item -ItemType Directory -Force $output | Out-Null
foreach ($stage in ('vert','frag')) {
    $source = Join-Path $root "shaders/gaussian.$stage"
    $target = Join-Path $output "gaussian.$stage.spv"
    & $glslc "-fshader-stage=$stage" $source -o $target
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation failed: $source" }
    & $spirvVal $target
    if ($LASTEXITCODE -ne 0) { throw "SPIR-V validation failed: $target" }
}
foreach ($name in @('raster.comp','present.vert','present.frag')) {
    $stage=($name -split '\.')[-1]
    $target=Join-Path $output "$name.spv"
    & $glslc "-fshader-stage=$stage" (Join-Path $root "shaders/$name") -o $target
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation failed: $name" }
    & $spirvVal $target
    if ($LASTEXITCODE -ne 0) { throw "SPIR-V validation failed: $name" }
}
