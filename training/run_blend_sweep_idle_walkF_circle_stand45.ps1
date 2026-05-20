$ErrorActionPreference = "Stop"

$root = "C:\Users\singerie\Documents\Cursor\stepper"
$logDir = Join-Path $root "training\runs\sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "blend_sweep_idle_walkF_circle_stand45_20260519.log"

$ae1 = "training\runs\20260517_161820_denoise_rootlook1_lat32_n0p05_e360\checkpoints\checkpoint_best.pt"
$ae2 = "training\runs\20260519_043759_compat_rootw16_yawbody_lat64_e180\checkpoints\checkpoint_best.pt"
$baseline = "training\runs\20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt\checkpoints\checkpoint_best_k32.pt"
$periodic = "training\runs\mini_datasets\idle_walkF_circle_stand45\periodic"
$nonperiodic = "training\runs\mini_datasets\idle_walkF_circle_stand45\nonperiodic"

Set-Location $root
"blend sweep started $(Get-Date -Format o)" | Tee-Object -FilePath $logPath

for ($i = 0; $i -lt 10; $i++) {
    $ae2w = $i / 9.0
    $ae1w = 1.0 - $ae2w
    $tag = "blend" + $i.ToString("00") + "_ae1_" + ([Math]::Round($ae1w * 100)).ToString("000") + "_ae2_" + ([Math]::Round($ae2w * 100)).ToString("000")
    "`n=== $tag ae1=$ae1w ae2=$ae2w $(Get-Date -Format o) ===" | Tee-Object -FilePath $logPath -Append
    python training\train_locomotion_ae_prior.py `
        --periodic-folder-path $periodic `
        --nonperiodic-folder-path $nonperiodic `
        --prior-checkpoint $ae1 `
        --prior-weight $ae1w `
        --extra-prior-checkpoint $ae2 `
        --extra-prior-weight $ae2w `
        --resume-checkpoint $baseline `
        --run-name "blend_sweep_idle_walkF_circle_stand45_$tag" `
        --hidden-dim 512 `
        --num-hidden-layers 2 `
        --learning-rate 0.00005 `
        --batch-size 64 `
        --training-loop agents `
        --agent-sampling random `
        --agent-batches-per-epoch 1 `
        --packed-agent-rollout `
        --agent-batch-clips 0 `
        --rollout-schedule 2,4,8,16,32 `
        --mixed-rollout-cohorts `
        --mixed-rollout-cohort-schedule 2,4,8,16,32 `
        --mixed-rollout-cohort-weights 5,15,20,30,40 `
        --curriculum-min-epochs 40 `
        --curriculum-max-epochs-per-stage 90 `
        --curriculum-stall-patience-epochs 35 `
        --max-epochs 420 `
        --stop-on-final-stall `
        --no-contact-physics-losses `
        --diagnostic-metrics-every-epochs 0 `
        --no-live-viewer `
        --no-visual-reporter 2>&1 | Tee-Object -FilePath $logPath -Append
}

"blend sweep finished $(Get-Date -Format o)" | Tee-Object -FilePath $logPath -Append
