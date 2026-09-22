<#
.SYNOPSIS
    Starts whisper.cpp's server for single-pass speech translation.

.DESCRIPTION
    Whisper's translate task turns any supported language into English in one
    model pass, so this replaces both the R2T2 server and the translator.

    IMPORTANT -- do not use a *turbo* model for this. large-v3-turbo was
    distilled with translation data excluded, so it transcribes well but
    translates badly. Use large-v3 or medium when -Translate is on.

    whisper.cpp publishes no official Windows Vulkan binary, so you either
    build it yourself or use a third-party Vulkan build. See the README.

.EXAMPLE
    .\scripts\run_whisper_server.ps1
    .\scripts\run_whisper_server.ps1 -Model medium -Port 8082
    .\scripts\run_whisper_server.ps1 -WhisperServer C:\tools\whisper\whisper-server.exe
#>

[CmdletBinding()]
param(
    [string]$WhisperServer = "whisper-server",
    # large-v3 / medium translate well; *-turbo does not.
    [ValidateSet("large-v3", "medium", "small", "base", "large-v3-turbo")]
    [string]$Model = "large-v3",
    [string]$Quant = "q5_0",
    [int]$Port = 8082,
    [int]$Threads = 8,
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"
$ModelDir = Join-Path $PSScriptRoot "..\models"
# small/base are only published as q5_1; medium/large only as q5_0.
if ($Model -in @("small", "base") -and $Quant -eq "q5_0") { $Quant = "q5_1" }
$ModelFile = Join-Path $ModelDir "ggml-$Model-$Quant.bin"

if ($Model -like "*turbo*") {
    Write-Warning @"
$Model is a turbo model. Turbo variants were distilled without translation
data and produce poor English output for the translate task. Prefer
-Model large-v3 (or medium) for single-pass translation, or run this model
with translation off:  python -m rtsubs --asr whispercpp --no-whisper-translate
"@
}

if (-not (Test-Path $ModelFile)) {
    Write-Error "Missing model: $ModelFile`nRun: .\scripts\download_models.ps1 -WhisperOnly -WhisperModel $Model"
}

if (-not (Get-Command $WhisperServer -ErrorAction SilentlyContinue)) {
    Write-Error @"
'$WhisperServer' not found on PATH.

whisper.cpp ships no official Windows Vulkan binary. Either:
  * build it:  cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release
               (needs the Vulkan SDK and MSVC; produces whisper-server.exe)
  * or use a prebuilt Vulkan build and pass its path:
               -WhisperServer C:\path\to\whisper-server.exe

A CPU build also works -- pass -NoGpu -- but expect it to be several times
slower than the Vulkan build on your RX 6700M.
"@
}

$serverArgs = @(
    "--model", $ModelFile
    "--host", "127.0.0.1"
    "--port", $Port
    "--threads", $Threads
)
if ($NoGpu) { $serverArgs += "--no-gpu" }

Write-Host "Starting whisper.cpp server on http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  model : ggml-$Model-$Quant.bin" -ForegroundColor DarkGray
Write-Host "  gpu   : $(if ($NoGpu) { 'off (CPU)' } else { 'on (Vulkan if built with it)' })" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Then run:  python -m rtsubs --asr whispercpp" -ForegroundColor Cyan
Write-Host ""

& $WhisperServer @serverArgs
