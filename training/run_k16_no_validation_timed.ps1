param(
    [string]$Prefix = ("k16_noval_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".tools\python310\python.exe"
$Train = Join-Path $Root "training\train_locomotion.py"
$Runs = Join-Path $Root "training\runs"
$TimingPath = Join-Path $Runs "$Prefix`_timing.csv"

Push-Location $Root
try {
    "stage,run_name,start_local,end_local,seconds" | Set-Content -LiteralPath $TimingPath

    function Run-Stage {
        param(
            [string]$Stage,
            [string]$Name,
            [string[]]$StageArgs
        )
        $RunDir = Join-Path $Runs $Name
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        $LogPath = Join-Path $RunDir "console.log"
        $ErrPath = Join-Path $RunDir "console.err.log"
        $Start = Get-Date
        $Sw = [System.Diagnostics.Stopwatch]::StartNew()
        $Process = Start-Process -FilePath $Python `
            -ArgumentList (@($Train) + $StageArgs) `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $LogPath `
            -RedirectStandardError $ErrPath `
            -NoNewWindow `
            -Wait `
            -PassThru
        $Exit = $Process.ExitCode
        $Sw.Stop()
        $End = Get-Date
        if ($Exit -ne 0) {
            Get-Content -LiteralPath $LogPath -Tail 40
            Get-Content -LiteralPath $ErrPath -Tail 40
            throw "Stage $Stage failed with exit code $Exit"
        }
        "$Stage,$Name,$($Start.ToString('s')),$($End.ToString('s')),$([Math]::Round($Sw.Elapsed.TotalSeconds, 3))" |
            Add-Content -LiteralPath $TimingPath
        Get-Content -LiteralPath $LogPath -Tail 8
    }

    $Stage1 = "${Prefix}_stage1_k1"
    $Stage2 = "${Prefix}_stage2_k8_16"
    $Stage3 = "${Prefix}_stage3_k16_polish"
    $Stage4 = "${Prefix}_stage4_k16_final"

    $Common = @(
        "--folder-path", ".\data\npz_final",
        "--device", "cuda",
        "--batch-size", "64",
        "--val-fraction", "0",
        "--no-validation",
        "--future-window-seconds", "0.25",
        "--no-compile",
        "--profile-timing",
        "--save-last-every-epochs", "1000",
        "--save-best-every-epochs", "0"
    )

    Run-Stage "stage1_k1" $Stage1 ($Common + @(
        "--run-name", $Stage1,
        "--rollout-schedule", "1",
        "--learning-rate", "0.0003",
        "--alpha4-end-effector-location", "80",
        "--alpha6-full-body-location", "10",
        "--max-epochs", "520",
        "--writer-flush-every-epochs", "20",
        "--curriculum-stall-patience-epochs", "0"
    ))

    Run-Stage "stage2_k8_16" $Stage2 ($Common + @(
        "--run-name", $Stage2,
        "--resume-checkpoint", ".\training\runs\$Stage1\checkpoints\checkpoint_best.pt",
        "--rollout-schedule", "8,16",
        "--learning-rate", "0.00002",
        "--alpha4-end-effector-location", "200",
        "--alpha6-full-body-location", "10",
        "--curriculum-threshold", "0.025",
        "--curriculum-min-epochs", "40",
        "--curriculum-max-epochs-per-stage", "180",
        "--curriculum-patience-epochs", "8",
        "--max-epochs", "360",
        "--writer-flush-every-epochs", "10",
        "--curriculum-stall-patience-epochs", "30",
        "--curriculum-min-delta", "0.00005"
    ))

    Run-Stage "stage3_k16_polish_5e-6" $Stage3 ($Common + @(
        "--run-name", $Stage3,
        "--resume-checkpoint", ".\training\runs\$Stage2\checkpoints\checkpoint_best.pt",
        "--rollout-schedule", "16",
        "--learning-rate", "0.000005",
        "--alpha4-end-effector-location", "300",
        "--alpha6-full-body-location", "10",
        "--max-epochs", "80",
        "--writer-flush-every-epochs", "10",
        "--curriculum-stall-patience-epochs", "25",
        "--curriculum-min-delta", "0.00001"
    ))

    Run-Stage "stage4_k16_final_2e-6" $Stage4 ($Common + @(
        "--run-name", $Stage4,
        "--resume-checkpoint", ".\training\runs\$Stage3\checkpoints\checkpoint_best.pt",
        "--rollout-schedule", "16",
        "--learning-rate", "0.000002",
        "--alpha4-end-effector-location", "300",
        "--alpha6-full-body-location", "10",
        "--max-epochs", "120",
        "--writer-flush-every-epochs", "10",
        "--curriculum-stall-patience-epochs", "45",
        "--curriculum-min-delta", "0.000005"
    ))

    Write-Output "timing_csv=$TimingPath"
    Get-Content -LiteralPath $TimingPath
}
finally {
    Pop-Location
}
