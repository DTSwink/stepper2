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

$ikRuns = Get-ChildItem -LiteralPath $runsRoot -Directory |
    Where-Object {
        $_.Name -like "*_ik_*" -and
        (Get-ChildItem -LiteralPath (Join-Path $_.FullName "tb") -Filter "events.out.tfevents*" -ErrorAction SilentlyContinue | Select-Object -First 1)
    } |
    Sort-Object Name

if ($ikRuns.Count -lt 1) {
    throw "No IK TensorBoard runs found under: $runsRoot"
}

$resolvedRunsRoot = (Resolve-Path -LiteralPath $runsRoot).Path
if (Test-Path -LiteralPath $stackRoot) {
    $resolvedStackRoot = (Resolve-Path -LiteralPath $stackRoot).Path
    if (-not $resolvedStackRoot.StartsWith($resolvedRunsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clear unexpected TensorBoard stack path: $resolvedStackRoot"
    }
    Remove-Item -LiteralPath $stackRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stackRoot | Out-Null

foreach ($run in $ikRuns) {
    $sourceTb = Join-Path $run.FullName "tb"
    $targetRun = Join-Path $stackRoot $run.Name
    New-Item -ItemType Directory -Path $targetRun | Out-Null
    Get-ChildItem -LiteralPath $sourceTb -Filter "events.out.tfevents*" |
        Copy-Item -Destination $targetRun -Force
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

Start-Sleep -Seconds 3

$logdir = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:6006/data/logdir").Content
Write-Host "TensorBoard: http://127.0.0.1:6006/"
Write-Host "Runs: $($ikRuns.Count)"
Write-Host "Logdir: $logdir"
