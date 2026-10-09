param(
    [switch]$Clean,
    [string]$Python = 'G:\Strata\Strata\.venv\Scripts\python.exe',
    [string]$CMake = 'G:\Strata\Strata\.venv\Scripts\cmake.exe',
    [string]$Ninja = 'G:\Strata\Strata\.venv\Scripts\ninja.exe',
    [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo 'build-safetensors'
# Import compiler environment only. No deployment scripts or model writes.
New-Item -ItemType Directory -Path $build -Force | Out-Null
$envScript = Join-Path $build 'compiler-env.cmd'
@("@call `"$VcVars`" >nul", '@if errorlevel 1 exit /b 1', '@set') | Set-Content -LiteralPath $envScript -Encoding ASCII
$environment = & $env:ComSpec /d /c $envScript
if ($LASTEXITCODE -ne 0) { throw 'vcvars64 failed' }
foreach ($line in $environment) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:VSLANG = '1033'
& $CMake -S (Join-Path $repo 'native_safetensors') -B $build -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$Ninja" "-DPython3_EXECUTABLE=$Python"
if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
if ($Clean) { & $CMake --build $build --clean-first } else { & $CMake --build $build }
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
& (Join-Path (Split-Path $CMake) 'ctest.exe') --test-dir $build --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
