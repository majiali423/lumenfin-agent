param(
    [Parameter(Mandatory = $true)]
    [string]$BackupDir
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -LiteralPath $BackupDir).Path
$manifestPath = Join-Path $resolved "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Backup manifest not found: $manifestPath"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($file in $manifest.files) {
    $path = Join-Path $resolved $file.name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Backup file is missing: $($file.name)"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne [string]$file.sha256) {
        throw "SHA256 mismatch: $($file.name)"
    }
}

$postgresDump = Join-Path $resolved "postgres.dump"
if (Test-Path -LiteralPath $postgresDump) {
    docker run --rm --read-only `
        -v "${resolved}:/backup:ro" `
        --entrypoint pg_restore `
        postgres:16@sha256:33f923b05f64ca54ac4401c01126a6b92afe839a0aa0a52bc5aeb5cc958e5f20 `
        --list /backup/postgres.dump | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "PostgreSQL dump catalog verification failed." }
}

foreach ($archive in Get-ChildItem -LiteralPath $resolved -Filter "*.tgz" -File) {
    docker run --rm --read-only `
        -v "${resolved}:/backup:ro" `
        python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 `
        tar -tzf "/backup/$($archive.Name)" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Archive verification failed: $($archive.Name)" }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($archive in Get-ChildItem -LiteralPath $resolved -Filter "*.zip" -File) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($archive.FullName)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName.Contains("..")) {
                throw "Unsafe ZIP entry in $($archive.Name): $($entry.FullName)"
            }
        }
    }
    finally {
        $zip.Dispose()
    }
}

Write-Output "Backup verification passed: $resolved"
