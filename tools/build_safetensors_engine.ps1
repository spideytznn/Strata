param(
    [int]$Jobs = 8,
    [string[]]$Targets = @('strata'),
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$OutputName = 'strata'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo 'build-native-engine'
$outputExe = Join-Path $build "$OutputName.exe"
foreach ($process in Get-CimInstance Win32_Process -Filter "Name='$OutputName.exe'") {
    if ($process.ExecutablePath -and [string]::Equals($process.ExecutablePath, $outputExe, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Engine is running at $outputExe. Use -OutputName with another name to stage this build."
    }
}
$bin = 'G:\Strata\Strata\.venv\Scripts'
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
New-Item -ItemType Directory -Path $build -Force | Out-Null
$envScript = Join-Path $build 'compiler-env.cmd'
@('@call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul', '@if errorlevel 1 exit /b 1', '@set') | Set-Content -LiteralPath $envScript -Encoding ASCII
$environment = & $env:ComSpec /d /c $envScript
if ($LASTEXITCODE -ne 0) { throw 'vcvars64 failed' }
foreach ($line in $environment) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:PATH = "$cuda\bin;$env:PATH"
$ErrorActionPreference = 'Continue' # PowerShell 5 treats native stderr warnings as ErrorRecords
& "$bin\cmake.exe" -S $repo -B $build -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$bin\ninja.exe" '-DSTRATA_ENABLE_CUDA=ON' '-DCMAKE_CUDA_ARCHITECTURES=120' "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe" '-DSTRATA_GGML_DIR=G:/Strata/Strata/third_party/llama.cpp' '-DSTRATA_NVFP4_TC=ON' '-DSTRATA_BUILD_TESTS=OFF' '-DSTRATA_BUILD_CONVERSATION_TESTS=ON' '-DCMAKE_CXX_FLAGS=/utf-8 /EHsc' "-DSTRATA_EXECUTABLE_NAME=$OutputName"
if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
& "$bin\cmake.exe" --build $build --target $Targets --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
