param([string]$Python)
$ErrorActionPreference = 'Stop'
$coreRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path (Split-Path -Parent $coreRoot) 'EvoScientist\.venv\Scripts\python.exe'
}
& $Python -c 'import sys; assert sys.version_info >= (3,12), "OpenHands requires Python 3.12+"'
if ($LASTEXITCODE -ne 0) { throw 'A Python 3.12+ interpreter is required' }
$environmentRoot = Join-Path $coreRoot '.module_envs\openhands'
& $Python -m venv $environmentRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not create OpenHands environment' }
$sdkPython = Join-Path $environmentRoot 'Scripts\python.exe'
& $sdkPython -m pip install 'openhands-sdk==1.62.0' 'openhands-tools==1.62.0'
if ($LASTEXITCODE -ne 0) { throw 'Could not install OpenHands packages' }
& $sdkPython -m pip freeze | Set-Content -LiteralPath (Join-Path $environmentRoot 'requirements.lock') -Encoding utf8
Write-Output "OpenHands environment: $sdkPython"
