$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing .venv; first run py -V:3.13 -m venv .venv" }
& $Python -m compileall -q (Join-Path $PSScriptRoot "..\src") (Join-Path $PSScriptRoot "..\tests") (Join-Path $PSScriptRoot "..\datasets_processing")
if ($LASTEXITCODE -ne 0) { throw "Python compile failed" }
& $Python -m pytest -q (Join-Path $PSScriptRoot "..\tests")
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency conflicts detected" }
Write-Host "Checks passed. Full pretrained XLM-R weights still require a real training smoke test."
