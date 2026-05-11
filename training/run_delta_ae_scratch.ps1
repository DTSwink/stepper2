param(
    [int]$HiddenDim = 512,
    [int]$LatentDim = 64,
    [int]$BatchSize = 64,
    [int]$K1WarmupEpochs = 900,
    [int]$K1PolishEpochs = 700,
    [int]$AutoregEpochs = 900,
    [int]$K1WarmupSeconds = 130,
    [int]$K1PolishSeconds = 120,
    [int]$AutoregSeconds = 260,
    [int]$SaveLiveEveryEpochs = 0,
    [switch]$LiveViewer,
    [string]$Prefix = ("delta_ae_scratch_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".tools\python310\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$Runs = Join-Path $Root "training\runs"
$TimingPath = Join-Path $Runs "$Prefix`_timing.csv"
$AETrain = Join-Path $Root "training\transition_autoencoder_deltas.py"
$ModelTrain = Join-Path $Root "training\train_locomotion_ae_prior_deltas.py"
$Visualize = Join-Path $Root "training\visualize_model.py"

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $Runs | Out-Null
    "stage,run_name,start_local,end_local,seconds" | Set-Content -LiteralPath $TimingPath

    function Run-Stage {
        param(
            [string]$Stage,
            [string]$Name,
            [string[]]$Args
        )
        $RunDir = Join-Path $Runs $Name
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        $LogPath = Join-Path $RunDir "console.log"
        $ErrPath = Join-Path $RunDir "console.err.log"
        $Start = Get-Date
        $Sw = [System.Diagnostics.Stopwatch]::StartNew()
        $Process = Start-Process -FilePath $Python `
            -ArgumentList $Args `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $LogPath `
            -RedirectStandardError $ErrPath `
            -NoNewWindow `
            -Wait `
            -PassThru
        $Sw.Stop()
        $End = Get-Date
        if ($Process.ExitCode -ne 0) {
            Get-Content -LiteralPath $LogPath -Tail 60
            Get-Content -LiteralPath $ErrPath -Tail 60
            throw "Stage $Stage failed with exit code $($Process.ExitCode)"
        }
        "$Stage,$Name,$($Start.ToString('s')),$($End.ToString('s')),$([Math]::Round($Sw.Elapsed.TotalSeconds, 3))" |
            Add-Content -LiteralPath $TimingPath
        Get-Content -LiteralPath $LogPath -Tail 12
    }

    $AERun = "${Prefix}_ae"
    $K1WarmupRun = "${Prefix}_k1_lr1e5"
    $K1PolishRun = "${Prefix}_k1_lr5e6"
    $AutoregRun = "${Prefix}_autoreg_k8"

    Run-Stage "transition_ae" $AERun @(
        $AETrain,
        "--folder-path", "data/npz_final",
        "--run-name", $AERun,
        "--latent-dim", "$LatentDim",
        "--hidden-dim", "$HiddenDim",
        "--learning-rate", "1e-3",
        "--max-epochs", "2000",
        "--target-loss-reduction", "0.995",
        "--stall-patience-epochs", "120",
        "--tier-eval-every-epochs", "25",
        "--device", "cuda"
    )

    $AECkpt = "training/runs/$AERun/checkpoints/checkpoint_best.pt"

    Run-Stage "model_k1_warmup" $K1WarmupRun @(
        $ModelTrain,
        "--folder-path", "data/npz_final",
        "--prior-checkpoint", $AECkpt,
        "--run-name", $K1WarmupRun,
        "--hidden-dim", "$HiddenDim",
        "--learning-rate", "1e-5",
        "--batch-size", "$BatchSize",
        "--rollout-schedule", "1",
        "--max-epochs", "$K1WarmupEpochs",
        "--max-train-seconds", "$K1WarmupSeconds",
        "--save-live-every-epochs", "0",
        "--best-metric", "joint_rmse",
        "--device", "cuda"
    )

    $K1WarmupCkpt = "training/runs/$K1WarmupRun/checkpoints/checkpoint_best.pt"

    Run-Stage "model_k1_polish" $K1PolishRun @(
        $ModelTrain,
        "--folder-path", "data/npz_final",
        "--prior-checkpoint", $AECkpt,
        "--resume-checkpoint", $K1WarmupCkpt,
        "--run-name", $K1PolishRun,
        "--hidden-dim", "$HiddenDim",
        "--learning-rate", "5e-6",
        "--batch-size", "$BatchSize",
        "--rollout-schedule", "1",
        "--max-epochs", "$K1PolishEpochs",
        "--max-train-seconds", "$K1PolishSeconds",
        "--save-live-every-epochs", "0",
        "--best-metric", "joint_rmse",
        "--device", "cuda"
    )

    $K1PolishCkpt = "training/runs/$K1PolishRun/checkpoints/checkpoint_best.pt"

    $AutoregArgs = @(
        $ModelTrain,
        "--folder-path", "data/npz_final",
        "--prior-checkpoint", $AECkpt,
        "--resume-checkpoint", $K1PolishCkpt,
        "--run-name", $AutoregRun,
        "--hidden-dim", "$HiddenDim",
        "--learning-rate", "5e-6",
        "--batch-size", "$BatchSize",
        "--rollout-schedule", "2,4,8",
        "--curriculum-max-epochs-per-stage", "240",
        "--curriculum-stall-patience-epochs", "80",
        "--curriculum-min-epochs", "30",
        "--max-epochs", "$AutoregEpochs",
        "--max-train-seconds", "$AutoregSeconds",
        "--save-live-every-epochs", "$SaveLiveEveryEpochs",
        "--best-metric", "joint_rmse",
        "--device", "cuda"
    )
    if ($LiveViewer) {
        $AutoregArgs += "--live-viewer"
    }
    Run-Stage "model_autoreg_k8" $AutoregRun $AutoregArgs

    $FinalCkpt = "training/runs/$AutoregRun/checkpoints/checkpoint_best.pt"
    $FinalHtml = "training/runs/model_comparisons/model_comparison.html"
    & $Python $Visualize --checkpoint-path $FinalCkpt --output-path $FinalHtml --device cuda

    Write-Output "ae_run=$AERun"
    Write-Output "k1_warmup_run=$K1WarmupRun"
    Write-Output "k1_polish_run=$K1PolishRun"
    Write-Output "autoreg_run=$AutoregRun"
    Write-Output "final_checkpoint=$FinalCkpt"
    Write-Output "final_viewer=$FinalHtml"
    Write-Output "timing_csv=$TimingPath"
    Get-Content -LiteralPath $TimingPath
}
finally {
    Pop-Location
}
