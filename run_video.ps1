param(
    [Parameter(Mandatory=$true)][string]$InputVideo,
    [string]$OutputVideo = 'output\enhanced.mp4'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    & $taskPython -m pip install -r requirements.txt
}
& $taskPython enhance_video.py --input $InputVideo --output $OutputVideo
if ($LASTEXITCODE -ne 0) { throw 'Video processing failed.' }
