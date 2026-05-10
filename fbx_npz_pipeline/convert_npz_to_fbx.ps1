param(
    [Parameter(Mandatory = $true)]
    [string]$InputNpz,

    [Parameter(Mandatory = $true)]
    [string]$TemplateFbx,

    [string]$OutputFbx = "",
    [string]$ReportJson = ""
)

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".tools\python310\python.exe"
$Converter = Join-Path $PSScriptRoot "npz_to_fbx.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Local Python was not found at $Python"
}
if (-not (Test-Path -LiteralPath $InputNpz)) {
    throw "Input NPZ was not found: $InputNpz"
}
if (-not (Test-Path -LiteralPath $TemplateFbx)) {
    throw "Template FBX was not found: $TemplateFbx"
}

$argsList = @($Converter, (Resolve-Path -LiteralPath $InputNpz).Path, "--template", (Resolve-Path -LiteralPath $TemplateFbx).Path)
if ($OutputFbx -ne "") {
    $argsList += @("-o", $OutputFbx)
}
if ($ReportJson -ne "") {
    $argsList += @("--report", $ReportJson)
}

& $Python @argsList
