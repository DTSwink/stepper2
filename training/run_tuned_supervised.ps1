$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".tools\python310\python.exe"
$Train = Join-Path $Root "training\train_locomotion.py"
$Runs = Join-Path $Root "training\runs"

function Run-Stage {
    param(
        [string]$Name,
        [string[]]$Args
    )
    $RunDir = Join-Path $Runs $Name
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    & $Python $Train @Args *> (Join-Path $RunDir "console.log")
    Get-Content (Join-Path $RunDir "console.log") -Tail 8
}

Run-Stage "tuned_stage1_k1_fast_overfit" @(
    "--folder-path", ".\data\npz_final",
    "--run-name", "tuned_stage1_k1_fast_overfit",
    "--device", "cuda",
    "--batch-size", "64",
    "--val-fraction", "0",
    "--future-window-seconds", "0.25",
    "--rollout-schedule", "1",
    "--learning-rate", "0.0003",
    "--alpha4-end-effector-location", "80",
    "--alpha6-full-body-location", "10",
    "--max-epochs", "520",
    "--no-compile",
    "--save-last-every-epochs", "1000",
    "--save-best-every-epochs", "0",
    "--writer-flush-every-epochs", "20",
    "--curriculum-stall-patience-epochs", "0"
)

Run-Stage "tuned_stage2_ar_8_16_32" @(
    "--folder-path", ".\data\npz_final",
    "--run-name", "tuned_stage2_ar_8_16_32",
    "--resume-checkpoint", ".\training\runs\tuned_stage1_k1_fast_overfit\checkpoints\checkpoint_best.pt",
    "--device", "cuda",
    "--batch-size", "64",
    "--val-fraction", "0",
    "--future-window-seconds", "0.25",
    "--rollout-schedule", "8,16,32",
    "--learning-rate", "0.00002",
    "--alpha4-end-effector-location", "200",
    "--alpha6-full-body-location", "10",
    "--curriculum-threshold", "0.025",
    "--curriculum-min-epochs", "40",
    "--curriculum-max-epochs-per-stage", "220",
    "--curriculum-patience-epochs", "8",
    "--max-epochs", "500",
    "--no-compile",
    "--save-last-every-epochs", "1000",
    "--save-best-every-epochs", "0",
    "--writer-flush-every-epochs", "10",
    "--curriculum-stall-patience-epochs", "40",
    "--curriculum-min-delta", "0.00005"
)

Run-Stage "tuned_stage3_k32_polish" @(
    "--folder-path", ".\data\npz_final",
    "--run-name", "tuned_stage3_k32_polish",
    "--resume-checkpoint", ".\training\runs\tuned_stage2_ar_8_16_32\checkpoints\checkpoint_best.pt",
    "--device", "cuda",
    "--batch-size", "64",
    "--val-fraction", "0",
    "--future-window-seconds", "0.25",
    "--rollout-schedule", "32",
    "--learning-rate", "0.000005",
    "--alpha4-end-effector-location", "300",
    "--alpha6-full-body-location", "10",
    "--max-epochs", "80",
    "--no-compile",
    "--save-last-every-epochs", "1000",
    "--save-best-every-epochs", "0",
    "--writer-flush-every-epochs", "10",
    "--curriculum-stall-patience-epochs", "25",
    "--curriculum-min-delta", "0.00001"
)

Run-Stage "tuned_stage4_k32_final" @(
    "--folder-path", ".\data\npz_final",
    "--run-name", "tuned_stage4_k32_final",
    "--resume-checkpoint", ".\training\runs\tuned_stage3_k32_polish\checkpoints\checkpoint_best.pt",
    "--device", "cuda",
    "--batch-size", "64",
    "--val-fraction", "0",
    "--future-window-seconds", "0.25",
    "--rollout-schedule", "32",
    "--learning-rate", "0.000002",
    "--alpha4-end-effector-location", "300",
    "--alpha6-full-body-location", "10",
    "--max-epochs", "120",
    "--no-compile",
    "--save-last-every-epochs", "1000",
    "--save-best-every-epochs", "0",
    "--writer-flush-every-epochs", "10",
    "--curriculum-stall-patience-epochs", "45",
    "--curriculum-min-delta", "0.000005"
)
