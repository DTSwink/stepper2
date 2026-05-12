$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.tools\python310\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
& $Python (Join-Path $PSScriptRoot "model_viewer_app.py")
