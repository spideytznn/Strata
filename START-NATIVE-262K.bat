@echo off
title Strata Safetensors - 262144 context - 8880
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-fast.json" -Port 8880
if errorlevel 1 pause
