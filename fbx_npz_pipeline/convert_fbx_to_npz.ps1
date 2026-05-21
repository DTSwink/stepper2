param(
    [Parameter(Mandatory = $true)]
    [string]$InputFbx,

    [string]$OutputNpz = "",
    [string]$ReportJson = "",
    [string]$OutputFinalNpz = ""
)

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".tools\python310\python.exe"
$Converter = Join-Path $PSScriptRoot "fbx_to_npz.py"
$Pruner = Join-Path $PSScriptRoot "prune_npz_skeleton.py"
$ModelBake = Join-Path $PSScriptRoot "make_model_reconstructable_npz.py"
$IkBake = Join-Path $PSScriptRoot "bake_ik_reconstructable_npz.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Local Python was not found at $Python"
}

if (-not (Test-Path -LiteralPath $InputFbx)) {
    throw "Input FBX was not found: $InputFbx"
}

$argsList = @($Converter, (Resolve-Path -LiteralPath $InputFbx).Path)
$ResolvedOutputNpz = ""
if ($OutputNpz -ne "") {
    $argsList += @("-o", $OutputNpz)
    $ResolvedOutputNpz = $OutputNpz
} else {
    $InputName = [System.IO.Path]::GetFileNameWithoutExtension($InputFbx)
    $ResolvedOutputNpz = Join-Path $ProjectRoot "data\npz\$InputName.npz"
}
if ($ReportJson -ne "") {
    $argsList += @("--report", $ReportJson)
}

& $Python @argsList

if ($OutputFinalNpz -ne "") {
    & $Python $Pruner (Resolve-Path -LiteralPath $ResolvedOutputNpz).Path -o $OutputFinalNpz
    & $Python $ModelBake (Resolve-Path -LiteralPath $OutputFinalNpz).Path -o $OutputFinalNpz
    & $Python $IkBake (Resolve-Path -LiteralPath $OutputFinalNpz).Path -o $OutputFinalNpz
} else {
    $FinalDir = Join-Path $ProjectRoot "data\npz_final"
    & $Python $Pruner (Resolve-Path -LiteralPath $ResolvedOutputNpz).Path -o $FinalDir
    & $Python $ModelBake $FinalDir -o $FinalDir
    & $Python $IkBake $FinalDir -o $FinalDir
}
