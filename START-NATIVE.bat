@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-quality.json"
if errorlevel 1 pause
