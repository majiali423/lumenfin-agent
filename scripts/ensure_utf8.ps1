# Run before demo/API on Windows PowerShell to avoid Chinese mojibake in terminal.
# Usage: . .\scripts\ensure_utf8.ps1

$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)

if ($IsWindows -or $env:OS -match "Windows") {
    chcp 65001 | Out-Null
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[utf-8] Console and Python I/O set to UTF-8"
