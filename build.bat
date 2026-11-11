@echo off
setlocal

where cl.exe >nul 2>nul
if errorlevel 1 (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)

if not "%CUDA_PATH%"=="" (
    set "CUDAToolkit_ROOT=%CUDA_PATH%"
    set "PATH=%CUDA_PATH%\bin;%PATH%"
)

if "%CUDA_ARCH%"=="" set "CUDA_ARCH=native"

cd /d "%~dp0"

cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%
if errorlevel 1 exit /b 1

cmake --build build -j 8
if errorlevel 1 exit /b 1

echo [Build] Successfully built build\lotus_rain_sim.exe
