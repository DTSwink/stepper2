param(
    [string]$FbxPath = "ue5/example_cascadeur",
    [string]$NpzFinalPath = "",
    [switch]$RebuildNpzFinal,
    [int]$HiddenDim = 512,
    [int]$LatentDim = 64,
    [double]$AEStdFloor = 0.01,
    [ValidateSet("mse", "huber")]
    [string]$AEScoreLoss = "huber",
    [int]$BatchSize = 64,
    [int]$K1Epochs = 800,
    [int]$AutoregEpochs = 260,
    [int]$K1Seconds = 140,
    [int]$AutoregSeconds = 180,
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
$EnsureNpzFinal = Join-Path $Root "fbx_npz_pipeline\ensure_npz_final.ps1"

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $Runs | Out-Null
    "stage,run_name,start_local,end_local,seconds" | Set-Content -LiteralPath $TimingPath

    if ($NpzFinalPath -ne "") {
        $DatasetPath = (Resolve-Path -LiteralPath $NpzFinalPath).Path
    }
    else {
        $EnsureArgs = @("-FbxPath", $FbxPath)
        if ($RebuildNpzFinal) {
            $EnsureArgs += "-Force"
        }
        $EnsureOutput = & $EnsureNpzFinal @EnsureArgs
        $EnsureOutput | ForEach-Object { Write-Output $_ }
        $DatasetLine = $EnsureOutput | Where-Object { $_ -like "NPZ_FINAL_DIR=*" } | Select-Object -Last 1
        if (-not $DatasetLine) {
            throw "Could not resolve npz_final folder from $FbxPath"
        }
        $DatasetPath = $DatasetLine.Substring("NPZ_FINAL_DIR=".Length)
    }
    $DatasetNpzs = @(Get-ChildItem -LiteralPath $DatasetPath -Filter *.npz -File | Sort-Object Name)
    if ($DatasetNpzs.Count -eq 0) {
        throw "No .npz files found in dataset folder: $DatasetPath"
    }
    $ViewerNpz = $DatasetNpzs[0].FullName
    Write-Output "dataset_npz_final=$DatasetPath"
    Write-Output "viewer_npz=$ViewerNpz"

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
        $CleanArgs = @($StageArgs | Where-Object { $null -ne $_ -and $_ -ne "" })
        $Process = Start-Process -FilePath $Python `
            -ArgumentList $CleanArgs `
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
    $K1Run = "${Prefix}_k1_ae"
    $AutoregRun = "${Prefix}_autoreg_k8"

    Run-Stage "transition_ae" $AERun @(
        $AETrain,
        "--folder-path", $DatasetPath,
        "--run-name", $AERun,
        "--latent-dim", "$LatentDim",
        "--hidden-dim", "$HiddenDim",
        "--std-floor", "$AEStdFloor",
        "--learning-rate", "1e-3",
        "--max-epochs", "2000",
        "--target-loss-reduction", "0.995",
        "--stall-patience-epochs", "120",
        "--tier-eval-every-epochs", "25",
        "--device", "cuda"
    )

    $AECkpt = "training/runs/$AERun/checkpoints/checkpoint_best.pt"

    Run-Stage "model_k1" $K1Run @(
        $ModelTrain,
        "--folder-path", $DatasetPath,
        "--prior-checkpoint", $AECkpt,
        "--run-name", $K1Run,
        "--hidden-dim", "$HiddenDim",
        "--learning-rate", "3e-6",
        "--batch-size", "$BatchSize",
        "--rollout-schedule", "1",
        "--max-epochs", "$K1Epochs",
        "--max-train-seconds", "$K1Seconds",
        "--save-live-every-epochs", "0",
        "--best-metric", "ae_score",
        "--ae-score-loss", $AEScoreLoss,
        "--no-contact-physics-losses",
        "--device", "cuda"
    )

    $K1Ckpt = "training/runs/$K1Run/checkpoints/checkpoint_best.pt"

    $AutoregArgs = @(
        $ModelTrain,
        "--folder-path", $DatasetPath,
        "--prior-checkpoint", $AECkpt,
        "--resume-checkpoint", $K1Ckpt,
        "--run-name", $AutoregRun,
        "--hidden-dim", "$HiddenDim",
        "--learning-rate", "3e-6",
        "--batch-size", "$BatchSize",
        "--rollout-schedule", "2,4,8",
        "--curriculum-max-epochs-per-stage", "80",
        "--curriculum-stall-patience-epochs", "40",
        "--curriculum-min-epochs", "30",
        "--max-epochs", "$AutoregEpochs",
        "--max-train-seconds", "$AutoregSeconds",
        "--save-live-every-epochs", "$SaveLiveEveryEpochs",
        "--live-npz-path", $ViewerNpz,
        "--best-metric", "ae_score",
        "--ae-score-loss", $AEScoreLoss,
        "--no-contact-physics-losses",
        "--device", "cuda"
    )
    if ($LiveViewer) {
        $AutoregArgs += "--live-viewer"
    }
    Run-Stage "model_autoreg_k8" $AutoregRun $AutoregArgs

    $FinalCkpt = "training/runs/$AutoregRun/checkpoints/checkpoint_best.pt"
    $FinalHtml = "training/runs/model_comparisons/model_comparison.html"
    & $Python $Visualize --npz-path $ViewerNpz --checkpoint-path $FinalCkpt --output-path $FinalHtml --device cuda

    Write-Output "fbx_path=$FbxPath"
    Write-Output "npz_final=$DatasetPath"
    Write-Output "ae_run=$AERun"
    Write-Output "k1_run=$K1Run"
    Write-Output "autoreg_run=$AutoregRun"
    Write-Output "final_checkpoint=$FinalCkpt"
    Write-Output "final_viewer=$FinalHtml"
    Write-Output "timing_csv=$TimingPath"
    Get-Content -LiteralPath $TimingPath
}
finally {
    Pop-Location
}
