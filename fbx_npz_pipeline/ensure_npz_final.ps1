param(
    [Parameter(Mandatory = $true)]
    [string]$FbxPath,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".tools\python310\python.exe"
$Converter = Join-Path $PSScriptRoot "fbx_to_npz.py"
$Pruner = Join-Path $PSScriptRoot "prune_npz_skeleton.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Local Python was not found at $Python"
}

if (-not (Test-Path -LiteralPath $FbxPath)) {
    throw "FBX path was not found: $FbxPath"
}

$Resolved = Resolve-Path -LiteralPath $FbxPath
$Item = Get-Item -LiteralPath $Resolved.Path
if ($Item.PSIsContainer) {
    $SourceDir = $Item.FullName
    $FbxFiles = @(Get-ChildItem -LiteralPath $SourceDir -File | Where-Object {
        $_.Extension -ieq ".fbx"
    } | Sort-Object Name)
}
else {
    if ($Item.Extension -ine ".fbx") {
        throw "FBX path must be a folder or a .fbx file: $($Item.FullName)"
    }
    $SourceDir = $Item.DirectoryName
    $FbxFiles = @($Item)
}

if ($FbxFiles.Count -eq 0) {
    throw "No .fbx files found in $SourceDir"
}

$RawDir = Join-Path $SourceDir "npz"
$FinalDir = Join-Path $SourceDir "npz_final"
$ReportDir = Join-Path $SourceDir "reports"
$FinalFiles = @()
if (Test-Path -LiteralPath $FinalDir) {
    $FinalFiles = @(Get-ChildItem -LiteralPath $FinalDir -Filter *.npz -File -ErrorAction SilentlyContinue)
}

if ($FinalFiles.Count -gt 0 -and -not $Force) {
    Write-Output "Using existing npz_final: $FinalDir"
    Write-Output "NPZ_FINAL_DIR=$FinalDir"
    return
}

New-Item -ItemType Directory -Force -Path $RawDir, $FinalDir, $ReportDir | Out-Null

foreach ($Fbx in $FbxFiles) {
    $RawNpz = Join-Path $RawDir ($Fbx.BaseName + ".npz")
    $Report = Join-Path $ReportDir ($Fbx.BaseName + ".json")
    Write-Output "Converting $($Fbx.Name) -> $RawNpz"
    & $Python $Converter $Fbx.FullName -o $RawNpz --report $Report
}

Write-Output "Pruning raw NPZ folder -> $FinalDir"
& $Python $Pruner $RawDir -o $FinalDir --report (Join-Path $ReportDir "npz_final")

Write-Output "NPZ_FINAL_DIR=$FinalDir"
