param(
    [switch]$DiagOnly,
    [switch]$PreviewOnly,
    [switch]$AudioRemuxOnly
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

# Auto-detect optional local CUDA installation if nvcc is not already in PATH
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
        $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
    } elseif (Test-Path "D:\soft\cuda_toolchain\cuda126\bin\nvcc.exe") {
        $env:PATH = "D:\soft\cuda_toolchain\cuda126\Library\bin;D:\soft\cuda_toolchain\cuda126\bin;" + $env:PATH
    }
}

$EarshotDir = if ($env:EARSHOT_DIR) { $env:EARSHOT_DIR } else { "d:\dev\opendev\earshot-main\earshot-main" }
$EarshotPy  = if ($env:EARSHOT_PYTHON) { $env:EARSHOT_PYTHON } else { Join-Path $EarshotDir ".venv\Scripts\python.exe" }
$QaPy       = if (Test-Path $EarshotPy) { $EarshotPy } else { "python" }

if (-not $AudioRemuxOnly) {
    Write-Host "=== [1/5] Building CUDA + C++20 Simulation (-O3 --use_fast_math) ==="
    cmd.exe /c "$ProjectDir\build.bat"
    if ($LASTEXITCODE -ne 0) { throw "Build failed!" }

    $Exe = "$ProjectDir\build\lotus_rain_sim.exe"

    Write-Host "=== [2/5] Running Fast GPU Diagnostic Test (--diag) ==="
    & $Exe --diag
    if ($LASTEXITCODE -ne 0) { throw "GPU diagnostic test failed!" }
    if ($DiagOnly) { exit 0 }

    if ($PreviewOnly) {
        Write-Host "=== [3/5] Running 6-Keyframe Full-Res Preview + Exporting GPU Impact Events (--preview) ==="
        & $Exe --preview --out-dir $ProjectDir
        if ($LASTEXITCODE -ne 0) { throw "Preview run failed!" }
        if (Test-Path $EarshotPy) {
            Push-Location $EarshotDir
            & $EarshotPy "$ProjectDir\synthesize_earshot_audio.py" --events "$ProjectDir\gpu_impact_events.bin" --out "$ProjectDir\lotus_rain_65536.wav"
            Pop-Location
        }
        & $QaPy "$ProjectDir\verify_qa.py" $ProjectDir
        exit $LASTEXITCODE
    }

    Write-Host "=== [3/5] Rendering Full 1920x1080 @ 60fps (1,200 Frames) + Coupled C++20 Audio ==="
    & $Exe --out-dir $ProjectDir
    if ($LASTEXITCODE -ne 0) { throw "Full cinema render failed!" }
}

if (Test-Path $EarshotPy) {
    Write-Host "=== [4/5] Optional Earshot Procedural Foil Audio Remux ==="
    Push-Location $EarshotDir
    & $EarshotPy "$ProjectDir\synthesize_earshot_audio.py" --events "$ProjectDir\gpu_impact_events.bin" --out "$ProjectDir\lotus_rain_65536.wav"
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "Earshot audio synthesis failed!" }
    Pop-Location

    $RemuxTmp = "$ProjectDir\lotus_rain_65536_remux.mp4"
    ffmpeg -y -loglevel error -i "$ProjectDir\lotus_rain_65536_cuda.mp4" -i "$ProjectDir\lotus_rain_65536.wav" `
        -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -ar 48000 -shortest $RemuxTmp
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg remux failed!" }
    Move-Item -Force $RemuxTmp "$ProjectDir\lotus_rain_65536_cuda.mp4"
} else {
    Write-Host "=== [4/5] Using Built-in Zero-Dependency C++20 Coupled Soundtrack ==="
}

Write-Host "=== [5/5] Running Visual & Audio Verification Gate ==="
& $QaPy "$ProjectDir\verify_qa.py" $ProjectDir
if ($LASTEXITCODE -ne 0) { throw "Verification gate failed!" }

Write-Host "=== ALL GATES PASSED: $ProjectDir\lotus_rain_65536_cuda.mp4 ==="
