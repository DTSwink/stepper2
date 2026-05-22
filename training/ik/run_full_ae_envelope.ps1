$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$runsRoot = Join-Path $repoRoot "training\runs"
$pythonExe = Join-Path $repoRoot ".tools\python310\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python runtime not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $runsRoot)) {
    New-Item -ItemType Directory -Path $runsRoot | Out-Null
}

$attempt = 0
$maxDelaySeconds = 120

while ($true) {
    $attempt += 1
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $runsRoot "overnight_ik_ae_envelope_${stamp}_attempt${attempt}.out.log"
    $err = Join-Path $runsRoot "overnight_ik_ae_envelope_${stamp}_attempt${attempt}.err.log"

    Write-Host "Starting IK full AE envelope run attempt $attempt"
    Write-Host "stdout: $out"
    Write-Host "stderr: $err"

    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList @("training\ik\train_full_ae_envelope.py") `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -PassThru `
        -Wait

    if ($process.ExitCode -eq 0) {
        Write-Host "IK full AE envelope run completed."
        exit 0
    }

    Write-Host "IK full AE envelope attempt $attempt crashed with exit code $($process.ExitCode)."
    try {
        & (Join-Path $scriptDir "launch_tensorboard_latest.ps1")
    } catch {
        Write-Host "TensorBoard refresh failed after crash: $($_.Exception.Message)"
    }

    $delay = [Math]::Min($maxDelaySeconds, 10 * $attempt)
    Write-Host "Restarting in $delay seconds; latest checkpoints will be reused when available."
    Start-Sleep -Seconds $delay
}
