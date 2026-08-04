# Phase 3.2B multi-process integration entrypoint (Windows).
param(
    [switch]$Keep,
    [switch]$SkipLoad,
    [switch]$SkipInfra
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$argsList = @()
if ($Keep) { $argsList += "--keep" }
if ($SkipLoad) { $argsList += "--skip-load" }
if ($SkipInfra) { $argsList += "--skip-infra" }

python scripts/run_phase32b_integration.py @argsList
exit $LASTEXITCODE
