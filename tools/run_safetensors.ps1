param(
    [Parameter(Mandatory=$true)][string]$Config,
    [int]$Port = 8097
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv-native\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create this project virtual environment and install requirements-native.txt first.'
}
$configPath = (Resolve-Path -LiteralPath $Config).Path
Push-Location -LiteralPath $repo
try {
    & $python -m serve.server --engine strata --config $configPath --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Native server exited with code $LASTEXITCODE" }
} finally { Pop-Location }
