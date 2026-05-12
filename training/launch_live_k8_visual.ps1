$ErrorActionPreference = "Stop"

$TrainingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $TrainingDir
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = "desktop_live_k8_$Timestamp"
$RunDir = Join-Path $ProjectRoot "training\runs\$RunName"
$Python = Join-Path $ProjectRoot ".tools\python310\python.exe"

if (!(Test-Path $Python)) {
    $Python = "python"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$TrainScript = Join-Path $ProjectRoot "training\train_locomotion.py"
$Stdout = Join-Path $RunDir "launcher_stdout.log"
$Stderr = Join-Path $RunDir "launcher_stderr.log"
$Arguments = @(
    $TrainScript,
    "--folder-path", "data/fbx/npz_final",
    "--run-name", $RunName,
    "--device", "cuda",
    "--training-loop", "agents",
    "--agent-sampling", "coverage",
    "--rollout-schedule", "1,2,4,8",
    "--curriculum-max-epochs-per-stage", "70",
    "--curriculum-stall-patience-epochs", "35",
    "--curriculum-min-delta", "1e-5",
    "--max-epochs", "320",
    "--batch-size", "64",
    "--learning-rate", "1e-4",
    "--lr-schedule", "adaptive_plateau",
    "--lr-min-factor", "0.05",
    "--lr-plateau-patience-epochs", "12",
    "--lr-plateau-factor", "0.7",
    "--lr-plateau-threshold", "0.001",
    "--save-last-every-epochs", "10",
    "--save-best-every-epochs", "10",
    "--writer-flush-every-epochs", "10",
    "--no-compile",
    "--live-viewer-start-visualizing"
)

Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr
