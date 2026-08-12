param(
    [string]$OutputRoot = "backups",
    [switch]$SkipDerivedVolumes
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path (Resolve-Path ".") (Join-Path $OutputRoot "production-$stamp")
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

$writers = @("lumenfin-api", "lumenfin-worker", "lumenfin-index-worker")
$volumeServices = @("redis", "etcd", "minio", "milvus")
$runningWriters = @()
$pausedServices = @()

function Get-ContainerId([string]$service) {
    $id = [string](docker compose ps -a -q $service | Select-Object -First 1)
    $id = $id.Trim()
    if (-not $id) {
        throw "Compose service '$service' has no container."
    }
    return $id
}

try {
    foreach ($service in $writers) {
        $id = [string](docker compose ps -a -q $service | Select-Object -First 1)
        $id = $id.Trim()
        if ($id -and (docker inspect --format "{{.State.Running}}" $id).Trim() -eq "true") {
            $runningWriters += $service
        }
    }
    if ($runningWriters.Count) {
        docker compose stop @runningWriters | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to stop application writers." }
    }

    $postgresId = Get-ContainerId "postgres"
    docker exec $postgresId sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f /tmp/lumenfin.dump'
    if ($LASTEXITCODE -ne 0) { throw "PostgreSQL dump failed." }
    docker cp "${postgresId}:/tmp/lumenfin.dump" (Join-Path $backupDir "postgres.dump") | Out-Null
    docker exec $postgresId rm -f /tmp/lumenfin.dump

    foreach ($directory in @("uploads", "outputs")) {
        if (Test-Path -LiteralPath $directory) {
            Compress-Archive -Path "$directory\*" -DestinationPath (Join-Path $backupDir "$directory.zip") -Force
        }
    }

    if (-not $SkipDerivedVolumes) {
        foreach ($service in $volumeServices) {
            $id = Get-ContainerId $service
            if ((docker inspect --format "{{.State.Running}}" $id).Trim() -eq "true") {
                docker compose pause $service | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "Failed to pause '$service'." }
                $pausedServices += $service
            }
        }

        $backupMount = (Resolve-Path $backupDir).Path
        foreach ($service in $volumeServices) {
            $id = Get-ContainerId $service
            $container = @(docker inspect $id | ConvertFrom-Json)[0]
            $volumeNames = @(
                $container.Mounts |
                    Where-Object { $_.Type -eq "volume" } |
                    ForEach-Object { $_.Name }
            )
            if (-not $volumeNames.Count) {
                throw "No named volume found for stateful service '$service'."
            }
            foreach ($volumeName in $volumeNames) {
                $archive = "$volumeName.tgz"
                docker run --rm --read-only `
                    -v "${volumeName}:/source:ro" `
                    -v "${backupMount}:/backup" `
                    python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 `
                    tar -C /source -czf "/backup/$archive" .
                if ($LASTEXITCODE -ne 0) { throw "Snapshot failed for volume '$volumeName'." }
            }
        }
    }

    $files = Get-ChildItem -LiteralPath $backupDir -File | ForEach-Object {
        [ordered]@{
            name = $_.Name
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
        }
    }
    [ordered]@{
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        project = "lumenfin-agent"
        files = @($files)
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backupDir "manifest.json") -Encoding UTF8

    Write-Output "Backup completed: $backupDir"
}
finally {
    $cleanupErrors = @()
    foreach ($service in $pausedServices) {
        try {
            docker compose unpause $service | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Failed to unpause '$service'." }
        }
        catch {
            $cleanupErrors += $_.Exception.Message
        }
    }
    if ($cleanupErrors.Count) {
        throw "Backup cleanup failed after attempting every unpause: $($cleanupErrors -join '; ')"
    }
    if ($runningWriters.Count) {
        $deadline = (Get-Date).AddSeconds(120)
        do {
            $unhealthy = @()
            foreach ($service in @("postgres", "redis", "etcd", "minio", "milvus")) {
                $id = Get-ContainerId $service
                $state = @(docker inspect $id | ConvertFrom-Json)[0].State
                if (-not $state.Running -or ($state.Health -and $state.Health.Status -ne "healthy")) {
                    $unhealthy += $service
                }
            }
            if ($unhealthy.Count -and (Get-Date) -lt $deadline) {
                Start-Sleep -Seconds 2
            }
        } while ($unhealthy.Count -and (Get-Date) -lt $deadline)
        if ($unhealthy.Count) {
            throw "Dependencies did not recover after backup: $($unhealthy -join ', ')"
        }
        docker compose start @runningWriters | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to restart application writers." }
    }
}
