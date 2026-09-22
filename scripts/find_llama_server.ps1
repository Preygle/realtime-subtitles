<#
    Dot-sourced by run_asr_server.ps1 and run_translator_server.ps1.

    winget installs llama.cpp into its own package folder and appends that
    folder to the *user* PATH. Any process that was already running at the
    time -- Explorer included, so anything double-clicked -- keeps the old
    PATH until the user signs out. Looking only at $env:Path therefore fails
    right after installation, so this also checks the registry's current
    PATH and the winget package folder directly.
#>

function Resolve-LlamaServer {
    param([string]$Name = "llama-server")

    # 1. An explicit path, or anything already on this process's PATH.
    $found = Get-Command $Name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }

    # 2. The PATH as it is stored now, which includes post-login changes.
    $fresh = @(
        [Environment]::GetEnvironmentVariable("Path", "User"),
        [Environment]::GetEnvironmentVariable("Path", "Machine")
    ) -join ";"
    foreach ($dir in ($fresh -split ";")) {
        if (-not $dir) { continue }
        $candidate = Join-Path ([Environment]::ExpandEnvironmentVariables($dir)) "$Name.exe"
        if (Test-Path $candidate) { return $candidate }
    }

    # 3. winget's own locations, in case PATH was never updated at all.
    $winget = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet"
    $link = Join-Path $winget "Links\$Name.exe"
    if (Test-Path $link) { return $link }
    $pkg = Get-ChildItem (Join-Path $winget "Packages") -Directory -Filter "ggml.llamacpp*" -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "$Name.exe" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    if ($pkg) { return $pkg }

    return $null
}
