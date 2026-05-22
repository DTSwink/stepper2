$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$runsRoot = Join-Path $repoRoot "training\runs"
$stackRoot = Join-Path $runsRoot "tensorboard_stack"
$pythonExe = Join-Path $repoRoot ".tools\python310\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python runtime not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $runsRoot)) {
    throw "Runs directory not found: $runsRoot"
}

$resolvedRunsRoot = (Resolve-Path -LiteralPath $runsRoot).Path
$maxRuns = 80

# Remove the old copied mirror if it exists. Deleting this directory is safe:
# old launcher versions created it under training/runs and copied event files
# into it; it is not a source run.
if (Test-Path -LiteralPath $stackRoot) {
    $resolvedStackRoot = (Resolve-Path -LiteralPath $stackRoot).Path
    if (-not $resolvedStackRoot.StartsWith($resolvedRunsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clear unexpected TensorBoard stack path: $resolvedStackRoot"
    }
    Remove-Item -LiteralPath $stackRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stackRoot | Out-Null

$eventRuns = Get-ChildItem -LiteralPath $resolvedRunsRoot -Directory |
    Where-Object {
        $_.Name -notin @("cache", "model_comparisons", "tensorboard_stack") -and
        $_.Name -notlike "_crashed*"
    } |
    ForEach-Object {
        $latestEvent = Get-ChildItem -LiteralPath $_.FullName -Recurse -Filter "events.out.tfevents*" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -ne $latestEvent) {
            [PSCustomObject]@{
                Name = $_.Name
                TbDir = $latestEvent.DirectoryName
                LastWriteTime = $latestEvent.LastWriteTime
            }
        }
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First $maxRuns

if ($eventRuns.Count -lt 1) {
    throw "No TensorBoard event files found under: $resolvedRunsRoot"
}

foreach ($run in $eventRuns) {
    $safeName = $run.Name.Replace(",", "_").Replace(":", "_")
    $linkPath = Join-Path $stackRoot $safeName
    New-Item -ItemType Junction -Path $linkPath -Target $run.TbDir | Out-Null
}

$listeners = Get-NetTCPConnection -LocalPort 6006 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique

foreach ($processId in $listeners) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

$tbInfo = Join-Path $env:TEMP ".tensorboard-info"
if (Test-Path -LiteralPath $tbInfo) {
    Remove-Item -LiteralPath $tbInfo -Recurse -Force
}

Start-Process -FilePath $pythonExe -ArgumentList @(
    "-m", "tensorboard.main",
    "--logdir", $stackRoot,
    "--host", "127.0.0.1",
    "--port", "6006",
    "--reload_interval", "2"
) -WindowStyle Hidden

$deadline = (Get-Date).AddSeconds(60)
do {
    Start-Sleep -Milliseconds 500
    try {
        $logdir = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:6006/data/logdir" -TimeoutSec 2).Content
        $runsJson = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:6006/data/runs" -TimeoutSec 2).Content
        $runs = $runsJson | ConvertFrom-Json
        $runCount = if ($null -eq $runs) { 0 } elseif ($runs -is [array]) { $runs.Length } else { @($runs).Count }
        Write-Host "TensorBoard: http://127.0.0.1:6006/"
        Write-Host "Runs: $runCount"
        Write-Host "Logdir: $logdir"
        exit 0
    } catch {
        if ((Get-Date) -ge $deadline) {
            throw
        }
    }
} while ($true)
