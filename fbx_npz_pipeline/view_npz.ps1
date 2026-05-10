param(
    [Parameter(Mandatory = $true)]
    [string]$InputNpz,

    [string]$OutputHtml = ""
)

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".tools\python310\python.exe"
$Viewer = Join-Path $PSScriptRoot "npz_to_html_viewer.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Local Python was not found at $Python"
}

if (-not (Test-Path -LiteralPath $InputNpz)) {
    throw "Input NPZ was not found: $InputNpz"
}

$argsList = @($Viewer, (Resolve-Path -LiteralPath $InputNpz).Path)
if ($OutputHtml -ne "") {
    $argsList += @("-o", $OutputHtml)
}

& $Python @argsList
