# Run this ONCE on the target build platform with network access, commit uv.lock.
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
py -V:3.13 -m pip install "uv>=0.10,<1"
if ($LASTEXITCODE -ne 0) { throw "uv install failed" }
py -V:3.13 -m uv lock --python 3.13
if ($LASTEXITCODE -ne 0) { throw "dependency lock resolution failed" }
py -V:3.13 -m uv sync --locked --extra dev --extra data --extra leakage
if ($LASTEXITCODE -ne 0) { throw "locked dependency installation failed" }
Write-Host "uv.lock generated and verified; commit it alongside pyproject.toml."
