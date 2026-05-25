param(
    [string]$Label = "full_supervised_from_temp_baseline_continuing",
    [string]$BaselineCheckpoint = "training\runs\20260522_212952_ik_full_supervised_from_temp_baseline_watchdog\checkpoints\20260522_212952_ik_full_supervised_from_temp_baseline_watchdog_last.pt",
    [string]$PeriodicFolder = "ue5\animations_omni_only_full\npz_final",
    [string]$NonperiodicFolder = "ue5\animations_transitions_only_full_trimmed\npz_final",
    [int]$TrainSteps = 100000,
    [bool]$DisableCudaGraph = $false,
    [switch]$StartIfMissing,
    [switch]$KillDuplicates
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
Set-Location $repo

$safeLabel = ($Label -replace "[^A-Za-z0-9_.-]", "_")
$mutexName = "Global\ik_supervised_watchdog_$safeLabel"
$mutex = [System.Threading.Mutex]::new($false, $mutexName)
$hasLock = $false

try {
    $hasLock = $mutex.WaitOne(0)
    if (-not $hasLock) {
        Write-Output "watchdog_status=busy label=$Label"
        exit 0
    }

    $pythonCmd = Get-Command python -ErrorAction Stop
    $python = $pythonCmd.Source
    $runsRoot = Join-Path $repo "training\runs"
    $logsRoot = $runsRoot
    $now = Get-Date -Format "yyyyMMdd_HHmmss"

    function Get-RunDirs {
        Get-ChildItem $runsRoot -Directory -Filter "*_ik_$Label" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending
    }

    function Get-LatestCheckpoint([string]$Tag) {
        foreach ($run in Get-RunDirs) {
            $ckptDir = Join-Path $run.FullName "checkpoints"
            $ckpt = Get-ChildItem $ckptDir -Filter "*_$Tag.pt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($ckpt) {
                return $ckpt.FullName
            }
        }
        return $null
    }

    function Get-TrainingProcesses {
        @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
            Where-Object {
                $cmd = [string]$_.CommandLine
                if ([string]::IsNullOrWhiteSpace($cmd)) {
                    $false
                } else {
                    $lower = $cmd.ToLowerInvariant()
                    $needle = $Label.ToLowerInvariant()
                    $hasLabel = $lower.Contains($needle)
                    $hasTrain = $lower.Contains("train.py")
                    $hasIkPath = $lower.Contains("training\ik") -or $lower.Contains("training/ik")
                    $hasLabel -and $hasTrain -and $hasIkPath
                }
            } |
            Sort-Object CreationDate)
    }

    $procs = @(Get-TrainingProcesses)
    if ($procs.Count -gt 1) {
        if ($KillDuplicates) {
            $keep = $procs[0]
            $dupes = $procs | Select-Object -Skip 1
            foreach ($proc in $dupes) {
                Stop-Process -Id $proc.ProcessId -Force
            }
            Write-Output "watchdog_status=deduped label=$Label kept_pid=$($keep.ProcessId) stopped=$($dupes.ProcessId -join ',')"
            $procs = @($keep)
        } else {
            Write-Output "watchdog_status=duplicate_processes label=$Label pids=$($procs.ProcessId -join ',')"
            exit 2
        }
    }

    $latest = Get-LatestCheckpoint "latest"
    $last = Get-LatestCheckpoint "last"
    $best = Get-LatestCheckpoint "best"

    if ($procs.Count -eq 1) {
        $run = Get-RunDirs | Select-Object -First 1
        $latestInfo = if ($latest) { $latest } else { "" }
        Write-Output "watchdog_status=running label=$Label pid=$($procs[0].ProcessId) run=$($run.FullName) latest=$latestInfo"
        exit 0
    }

    if (-not $StartIfMissing) {
        Write-Output "watchdog_status=not_running label=$Label latest=$latest last=$last best=$best"
        exit 1
    }

    $init = $latest
    if (-not $init) {
        $init = $last
    }
    if (-not $init) {
        $init = (Resolve-Path $BaselineCheckpoint).Path
    }

    $outLog = Join-Path $logsRoot "watch_${safeLabel}_${now}.out.log"
    $errLog = Join-Path $logsRoot "watch_${safeLabel}_${now}.err.log"
    $args = @(
        "-u",
        "training\ik\train.py",
        "--run-label", $Label,
        "--periodic-folder", $PeriodicFolder,
        "--nonperiodic-folder", $NonperiodicFolder,
        "--init-checkpoint", $init,
        "--resume-step-from-checkpoint",
        "--load-optimizer",
        "--train-steps", ([string]$TrainSteps)
    )
    if ($DisableCudaGraph) {
        throw "DisableCudaGraph is forbidden; supervised IK training is CUDA-graph-only."
    }

    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $args `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -PassThru

    Write-Output "watchdog_status=started label=$Label pid=$($proc.Id) init=$init out=$outLog err=$errLog"
} finally {
    if ($hasLock) {
        $mutex.ReleaseMutex() | Out-Null
    }
    $mutex.Dispose()
}
