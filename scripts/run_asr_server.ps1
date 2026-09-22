<#
.SYNOPSIS
    Starts llama-server with Confucius4-R2T2 for streaming ASR.

.DESCRIPTION
    Vulkan is the backend to use on an RX 6700M under Windows: ROCm/HIP has no
    stable RDNA2 Windows support, and DirectML is both slower and not a
    llama.cpp backend at all. A Vulkan build of llama.cpp needs no toolkit
    beyond the normal Radeon driver.

    The mmproj file carries the audio encoder -- without it llama-server loads
    the text decoder only and /v1/audio/transcriptions will fail.

.EXAMPLE
    .\scripts\run_asr_server.ps1
    .\scripts\run_asr_server.ps1 -Quant Q4_K_M -Port 8090 -ContextSize 4096
#>

[CmdletBinding()]
param(
    [string]$LlamaServer = "llama-server",
    [string]$Quant = "Q8_0",
    [string]$MmprojQuant = "f16",
    [int]$Port = 8090,
    [int]$ContextSize = 4096,
    [int]$GpuLayers = 99,
    [string]$Device = "Vulkan0",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$ModelDir = Join-Path $PSScriptRoot "..\models"

$Model = Join-Path $ModelDir "Confucius4-R2T2-$Quant.gguf"
$Mmproj = Join-Path $ModelDir "mmproj-Confucius4-R2T2-$MmprojQuant.gguf"

foreach ($pair in @(@{P = $Model; N = "model" }, @{P = $Mmproj; N = "mmproj" })) {
    if (-not (Test-Path $pair.P)) {
        Write-Error "Missing $($pair.N): $($pair.P)`nRun: .\scripts\download_models.ps1 -AsrOnly"
    }
}

. (Join-Path $PSScriptRoot "find_llama_server.ps1")
$resolved = Resolve-LlamaServer $LlamaServer
if ($resolved) {
    $LlamaServer = $resolved
}
else {
    Write-Error @"
'$LlamaServer' not found on PATH.

Install a Vulkan build of llama.cpp:
  winget install ggml.llamacpp
or download the llama-*-bin-win-vulkan-x64.zip release from
  https://github.com/ggml-org/llama.cpp/releases
and either add it to PATH or pass -LlamaServer C:\path\to\llama-server.exe
"@
}

$serverArgs = @(
    "--model", $Model
    "--mmproj", $Mmproj
    "--port", $Port
    "--host", "127.0.0.1"
    "--ctx-size", $ContextSize
    # Audio segments arrive one at a time; a deep batch only adds latency.
    "--batch-size", "512"
    "--no-webui"
)

if ($Cpu) {
    $serverArgs += @("--gpu-layers", "0")
    Write-Host "Running on CPU (slow -- expect several seconds per segment)" -ForegroundColor Yellow
}
else {
    $serverArgs += @("--gpu-layers", $GpuLayers, "--device", $Device)
}

Write-Host "Starting R2T2 ASR server on http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  model  : $(Split-Path -Leaf $Model)" -ForegroundColor DarkGray
Write-Host "  mmproj : $(Split-Path -Leaf $Mmproj)" -ForegroundColor DarkGray
Write-Host "  device : $(if ($Cpu) { 'CPU' } else { $Device })" -ForegroundColor DarkGray
Write-Host ""

& $LlamaServer @serverArgs
