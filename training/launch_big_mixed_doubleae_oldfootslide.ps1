$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".tools\python310\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = "${stamp}_big_mixedk_doubleae_oldfootslide_from_k32"
$logDir = Join-Path $repoRoot "training\runs\launch_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$outPath = Join-Path $logDir "${runName}.out.log"
$errPath = Join-Path $logDir "${runName}.err.log"

$args = @(
    "training/train_locomotion_ae_prior.py",
    "--periodic-folder-path", "ue5/animations_omni_only_full/npz_final",
    "--nonperiodic-folder-path", "ue5/animations_transitions_only_full_trimmed/npz_final",
    "--prior-checkpoint", "training/runs/20260517_161820_denoise_rootlook1_lat32_n0p05_e360/checkpoints/checkpoint_best.pt",
    "--prior-weight", "1.0",
    "--extra-prior-checkpoint", "training/runs/20260517_184310_denoise_rootlook16_fullae_lat32_dampedcompat035_n0p05_e300/checkpoints/checkpoint_best.pt",
    "--extra-prior-weight", "0.30",
    "--compatibility-score-weight", "0.05",
    "--resume-checkpoint", "training/runs/20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt",
    "--run-name", $runName,
    "--device", "cuda",
    "--hidden-dim", "512",
    "--num-hidden-layers", "2",
    "--learning-rate", "1e-4",
    "--batch-size", "256",
    "--max-epochs", "10000000",
    "--rollout-schedule", "2,4,8,16,32,64",
    "--initial-rollout-k", "64",
    "--mixed-rollout-cohorts",
    "--mixed-rollout-cohort-schedule", "2,4,8,16,32,64",
    "--mixed-rollout-cohort-weights", "5,9,14,18,23,31",
    "--training-loop", "agents",
    "--agent-sampling", "random",
    "--agent-batch-clips", "0",
    "--packed-agent-rollout",
    "--agent-batches-per-epoch", "1",
    "--gradient-accumulation-batches", "1",
    "--periodic-sampling-weight", "2",
    "--nonperiodic-sampling-weight", "1",
    "--agent-min-cohort-steps", "2",
    "--no-contact-physics-losses",
    "--simple-footslide-loss-weight", "0.28",
    "--simple-footslide-threshold-mps", "0.2135299310088158",
    "--turn-idle-footslide-tolerance-divisor", "20",
    "--foot-yaw-loss-weight", "0",
    "--motion-floor-loss-weight", "0",
    "--no-support-envelope",
    "--diagnostic-metrics-every-epochs", "0",
    "--save-live-every-epochs", "20",
    "--no-live-viewer",
    "--no-visual-reporter",
    "--no-compile"
)

$process = Start-Process -FilePath $python `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $outPath `
    -RedirectStandardError $errPath `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started $runName"
Write-Host "PID $($process.Id)"
Write-Host "stdout $outPath"
Write-Host "stderr $errPath"
