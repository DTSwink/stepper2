$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $ProjectRoot ".tools\python310\python.exe"
if (!(Test-Path $Python)) {
    $Python = "python"
}

& $Python (Join-Path $ProjectRoot "training\ik\kaggle_prepare.py") --upload --push-kernel @args
