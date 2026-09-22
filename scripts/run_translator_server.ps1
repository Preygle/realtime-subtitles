<#
.SYNOPSIS
    Starts a second llama-server instance for translation.

.DESCRIPTION
    Runs on its own port so the ASR model stays resident in VRAM. A 4B
    Q4_K_M translator (~2.5 GB) next to R2T2 Q8_0 + mmproj (~2.0 GB) leaves
    roughly half of a 10 GB card free for context and the display.

    If VRAM gets tight, either drop the ASR model to Q4_K_M or switch the app
    to the NLLB translator, which runs entirely on the CPU.

.EXAMPLE
    .\scripts\run_translator_server.ps1
    .\scripts\run_translator_server.ps1 -GpuLayers 20   # partial offload
#>

[CmdletBinding()]
param(
    [string]$LlamaServer = "llama-server",
    [string]$ModelFile = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    [int]$Port = 8081,
    [int]$ContextSize = 4096,
    [int]$GpuLayers = 99,
    [string]$Device = "Vulkan0",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$ModelDir = Join-Path $PSScriptRoot "..\models"
$Model = Join-Path $ModelDir $ModelFile

if (-not (Test-Path $Model)) {
    Write-Error "Missing translator model: $Model`nRun: .\scripts\download_models.ps1 -TranslatorOnly"
}
. (Join-Path $PSScriptRoot "find_llama_server.ps1")
$resolved = Resolve-LlamaServer $LlamaServer
if ($resolved) {
    $LlamaServer = $resolved
}
else {
    Write-Error "'$LlamaServer' not found on PATH. See run_asr_server.ps1 for install notes."
}

$serverArgs = @(
    "--model", $Model
    "--port", $Port
    "--host", "127.0.0.1"
    "--ctx-size", $ContextSize
    # Subtitle lines are short; a large batch wastes VRAM here.
    "--batch-size", "256"
    "--no-webui"
)

if ($Cpu) {
    $serverArgs += @("--gpu-layers", "0")
}
else {
    $serverArgs += @("--gpu-layers", $GpuLayers, "--device", $Device)
}

Write-Host "Starting translator server on http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  model  : $ModelFile" -ForegroundColor DarkGray
Write-Host "  device : $(if ($Cpu) { 'CPU' } else { $Device })" -ForegroundColor DarkGray
Write-Host ""

& $LlamaServer @serverArgs
