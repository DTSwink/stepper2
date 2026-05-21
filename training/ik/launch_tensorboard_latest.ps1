$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$runsRoot = Join-Path $repoRoot "training\runs"
$pythonExe = Join-Path $repoRoot ".tools\python310\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python runtime not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $runsRoot)) {
    throw "Runs directory not found: $runsRoot"
}

$latestRun = Get-ChildItem -LiteralPath $runsRoot -Directory |
    Where-Object {
        $_.Name -like "*_ik_*" -and
        (Get-ChildItem -LiteralPath (Join-Path $_.FullName "tb") -Filter "events.out.tfevents*" -ErrorAction SilentlyContinue | Select-Object -First 1)
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $latestRun) {
    throw "No IK TensorBoard runs found under: $runsRoot"
}

$tbDir = Join-Path $latestRun.FullName "tb"

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
    "--logdir", $tbDir,
    "--host", "127.0.0.1",
    "--port", "6006",
    "--reload_interval", "2"
) -WindowStyle Hidden

Start-Sleep -Seconds 3

$logdir = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:6006/data/logdir").Content
Write-Host "TensorBoard: http://127.0.0.1:6006/"
Write-Host "Run: $($latestRun.Name)"
Write-Host "Logdir: $logdir"
