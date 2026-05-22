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

# TensorBoard must watch the real tb folders, not copied event files.
# It also must not recursively scan all of training/runs, because that tree
# contains archives and large nested run dumps. --logdir_spec gives live,
# top-level IK runs without the stale mirror or the deep crawl.
$ikRuns = Get-ChildItem -LiteralPath $resolvedRunsRoot -Directory |
    Where-Object {
        $_.Name -like "*_ik_*" -and
        (Get-ChildItem -LiteralPath (Join-Path $_.FullName "tb") -Filter "events.out.tfevents*" -ErrorAction SilentlyContinue | Select-Object -First 1)
    } |
    Sort-Object Name

if ($ikRuns.Count -lt 1) {
    throw "No top-level IK TensorBoard runs found under: $resolvedRunsRoot"
}

$logdirSpec = ($ikRuns | ForEach-Object {
    $runName = $_.Name.Replace(",", "_").Replace(":", "_")
    $tbDir = Join-Path $_.FullName "tb"
    "${runName}:$tbDir"
}) -join ","

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
    "--logdir_spec", $logdirSpec,
    "--host", "127.0.0.1",
    "--port", "6006",
    "--reload_interval", "2"
) -WindowStyle Hidden

$deadline = (Get-Date).AddSeconds(20)
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
