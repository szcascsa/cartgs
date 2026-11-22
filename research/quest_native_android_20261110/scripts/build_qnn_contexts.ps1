param(
    [Parameter(Mandatory=$true)][string]$QnnSdkRoot,
    [string]$OnnxDirectory = "$(Split-Path -Parent $PSScriptRoot)/models/onnx"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$converter = Join-Path $QnnSdkRoot 'bin/x86_64-linux-clang/qnn-onnx-converter'
$modelGenerator = Join-Path $QnnSdkRoot 'bin/x86_64-linux-clang/qnn-model-lib-generator'
$contextGenerator = Join-Path $QnnSdkRoot 'bin/x86_64-linux-clang/qnn-context-binary-generator'
foreach ($tool in ($converter,$modelGenerator,$contextGenerator)) {
    if (!(Test-Path $tool)) { throw "Required QAIRT/QNN tool is missing: $tool" }
}
throw @'
QAIRT/QNN SDK was not available during implementation, so the version-specific HTP V69 context command has not been validated.
Do not remove this stop until the installed SDK's converter, model-library and context-generator arguments are verified on SoC model 53.
Producing unvalidated .bin files would risk CPU/GPU fallback or a context ABI mismatch.
'@
