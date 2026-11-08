$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    & $taskPython -m pip install -r requirements.txt
}
& $taskPython make_demo.py
if ($LASTEXITCODE -ne 0) { throw 'Demo generation failed; see README for data download.' }
