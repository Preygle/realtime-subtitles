<#
.SYNOPSIS
    Downloads the model weights this app needs.

.DESCRIPTION
    Nothing here runs automatically -- invoke it when you are ready to pull
    several GB. Each component can be fetched on its own:

        .\scripts\download_models.ps1 -VadOnly       # ~2 MB, do this first
        .\scripts\download_models.ps1 -AsrOnly       # ~2 GB
        .\scripts\download_models.ps1 -TranslatorOnly # ~2.5 GB
        .\scripts\download_models.ps1                # everything above
        .\scripts\download_models.ps1 -WhisperOnly   # ~1.1 GB, alternative

    -WhisperOnly fetches a Whisper model for the single-pass backend,
    which replaces the ASR+translator pair rather than adding to it, so
    it is not part of "everything".

    Sizes assume the default quantisations. With an RX 6700M (10 GB VRAM) the
    Q8_0 ASR model plus a Q4_K_M 4B translator leaves plenty of headroom.

.NOTES
    Verify the translator repo/file names before the first run; GGUF
    re-uploaders rename files more often than the official repos do.
#>

[CmdletBinding()]
param(
    [switch]$VadOnly,
    [switch]$AsrOnly,
    [switch]$TranslatorOnly,
    [switch]$WhisperOnly,
    # large-v3 and medium translate well; turbo variants do not.
    [ValidateSet("large-v3", "medium", "small", "base", "large-v3-turbo")]
    [string]$WhisperModel = "large-v3",
    [string]$WhisperQuant = "q5_0",
    [string]$AsrQuant = "Q8_0",              # Q8_0 (1.7 GB) or Q4_K_M (1.0 GB)
    [string]$MmprojQuant = "f16",            # f16 (0.6 GB) or Q8_0 (0.3 GB)
    [string]$TranslatorRepo = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    [string]$TranslatorFile = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

$ErrorActionPreference = "Stop"
$ModelDir = Join-Path $PSScriptRoot "..\models" | Resolve-Path -ErrorAction SilentlyContinue
if (-not $ModelDir) {
    $ModelDir = Join-Path $PSScriptRoot "..\models"
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
}

function Get-File {
    param([string]$Url, [string]$Destination, [string]$Label)

    if (Test-Path $Destination) {
        $sizeMb = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
        Write-Host "  [skip] $Label already present ($sizeMb MB)" -ForegroundColor DarkGray
        return
    }
    Write-Host "  [get ] $Label" -ForegroundColor Cyan
    Write-Host "         $Url" -ForegroundColor DarkGray

    # A partial file must not look like a finished one on the next run.
    $temp = "$Destination.partial"
    # curl draws its progress bar on stderr. Under ErrorActionPreference=Stop,
    # Windows PowerShell 5.1 turns any native stderr line into a terminating
    # error, so relax it for the call and judge success by the exit code.
    #
    # On a slow or flaky link the connection can drop mid-transfer (curl exit
    # 56), which --retry alone does not cover. Each attempt resumes from the
    # bytes already in the .partial file, so retrying never loses progress.
    $maxAttempts = 100
    $code = -1
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 5 `
                --speed-limit 1024 --speed-time 60 `
                -C - --progress-bar -o $temp $Url
            $code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previous
        }
        # 0 = done. 33 = server refused the range because the file is already
        # complete, which also means done.
        if ($code -eq 0 -or $code -eq 33) { break }
        $have = if (Test-Path $temp) { [math]::Round((Get-Item $temp).Length / 1MB, 1) } else { 0 }
        Write-Host "  [retry] curl exit $code, $have MB so far; resuming ($attempt/$maxAttempts)" -ForegroundColor DarkYellow
        Start-Sleep -Seconds 10
    }
    if ($code -ne 0 -and $code -ne 33) {
        throw "Failed to download $Label after $maxAttempts attempts (curl exit code $code)"
    }
    Move-Item -Force $temp $Destination
}

# Whisper is an alternative to the ASR+translator pair, not part of the
# default set, so -WhisperOnly is opt-in and excluded from "everything".
$doAll = -not ($VadOnly -or $AsrOnly -or $TranslatorOnly -or $WhisperOnly)

# ---------------------------------------------------------------- VAD
if ($doAll -or $VadOnly) {
    Write-Host "`nSilero VAD (~2 MB)" -ForegroundColor Yellow
    Get-File `
        -Url "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" `
        -Destination (Join-Path $ModelDir "silero_vad.onnx") `
        -Label "silero_vad.onnx"
    # onnxruntime-directml also provides the CPU provider Silero uses; never
    # install both packages side by side, they overwrite each other.
    Write-Host "  Requires onnxruntime (or onnxruntime-directml, which also works)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- ASR
if ($doAll -or $AsrOnly) {
    Write-Host "`nConfucius4-R2T2 GGUF (ASR)" -ForegroundColor Yellow
    $repo = "https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF/resolve/main"

    Get-File -Url "$repo/Confucius4-R2T2-$AsrQuant.gguf" `
        -Destination (Join-Path $ModelDir "Confucius4-R2T2-$AsrQuant.gguf") `
        -Label "Confucius4-R2T2-$AsrQuant.gguf"

    # The mmproj holds the audio encoder; llama.cpp cannot do ASR without it.
    Get-File -Url "$repo/mmproj-Confucius4-R2T2-$MmprojQuant.gguf" `
        -Destination (Join-Path $ModelDir "mmproj-Confucius4-R2T2-$MmprojQuant.gguf") `
        -Label "mmproj-Confucius4-R2T2-$MmprojQuant.gguf"
}

# --------------------------------------------------------- Translator
if ($doAll -or $TranslatorOnly) {
    Write-Host "`nTranslator LLM GGUF" -ForegroundColor Yellow
    Get-File -Url "https://huggingface.co/$TranslatorRepo/resolve/main/$TranslatorFile" `
        -Destination (Join-Path $ModelDir $TranslatorFile) `
        -Label $TranslatorFile
}

# ------------------------------------------------------------- Whisper
if ($WhisperOnly) {
    Write-Host "`nWhisper GGML (single-pass speech translation)" -ForegroundColor Yellow
    if ($WhisperModel -like "*turbo*") {
        Write-Warning "Turbo models translate poorly; prefer large-v3 or medium."
    }
    # small/base are only published as q5_1; medium/large only as q5_0.
    if ($WhisperModel -in @("small", "base") -and $WhisperQuant -eq "q5_0") {
        $WhisperQuant = "q5_1"
    }
    $name = "ggml-$WhisperModel-$WhisperQuant.bin"
    Get-File -Url "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$name" `
        -Destination (Join-Path $ModelDir $name) `
        -Label $name
    Write-Host "  Next: run_whisper_server.ps1 -Model $WhisperModel (in scripts/)" -ForegroundColor DarkGray
}

Write-Host "`nModels in $ModelDir :" -ForegroundColor Green
Get-ChildItem $ModelDir -File |
    Sort-Object Length -Descending |
    Format-Table Name, @{N = "Size"; E = { "{0,8:N1} MB" -f ($_.Length / 1MB) } } -AutoSize

Write-Host "Next: .\scripts\run_asr_server.ps1  (and run_translator_server.ps1)" -ForegroundColor Green
