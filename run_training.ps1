param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".tools\python310\python.exe"
if (!(Test-Path $Python)) {
    $Python = "python"
}
$Runner = Join-Path $Root "training\ik\train.py"

& $Python $Runner @Args
